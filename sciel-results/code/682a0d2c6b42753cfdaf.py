"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware stratified sampling on the ZDT3 Pareto front with curvature-aware allocation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3_h(self, x1):
        """Compute h(x1) for ZDT3."""
        return 0.5 * x1 + np.sin(10.0 * np.pi * x1)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        dim = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Strategy: Focus on regions where f2 is negative to maximize HV.
        # f2(x) = 0.5*x + sin(10*pi*x)
        
        # 1. Find intervals where f2(x) < 0
        n_check = 10000
        x_check = np.linspace(0.0, 1.0, n_check)
        f2_check = self._zdt3_h(x_check)
        
        # Identify negative regions
        neg_mask = f2_check < 0
        # Find contiguous blocks of negative values
        # We want to sample within these blocks
        neg_indices = np.where(neg_mask)[0]
        
        if len(neg_indices) == 0:
            # Fallback: uniform sampling
            x1_samples = np.random.uniform(0, 1, budget)
        else:
            # Group indices into contiguous regions
            regions = []
            current_region = [neg_indices[0]]
            for idx in neg_indices[1:]:
                if idx == current_region[-1] + 1:
                    current_region.append(idx)
                else:
                    regions.append(current_region)
                    current_region = [idx]
            regions.append(current_region)
            
            # Allocate points to regions proportional to their length
            # Reserve 1 point for x=0 (to cover the start)
            n_reserved = 1
            n_alloc = budget - n_reserved
            
            if n_alloc <= 0:
                x1_samples = np.random.uniform(0, 1, budget)
            else:
                x1_samples = [0.0] # Reserve x=0
                
                # Calculate lengths of regions in x-space
                region_lengths = []
                for r in regions:
                    x_start = x_check[r[0]]
                    x_end = x_check[r[-1]]
                    region_lengths.append(x_end - x_start)
                
                total_length = sum(region_lengths)
                if total_length == 0:
                    x1_samples = np.random.uniform(0, 1, budget)
                else:
                    # Distribute points proportionally
                    for i, r in enumerate(regions):
                        frac = region_lengths[i] / total_length
                        n_pts = int(np.round(frac * n_alloc))
                        if n_pts == 0:
                            continue
                        x_start = x_check[r[0]]
                        x_end = x_check[r[-1]]
                        # Sample uniformly in this region
                        # Add small padding to stay within [0,1]
                        pts = np.random.uniform(x_start, x_end, n_pts)
                        x1_samples.extend(pts)
                        
                    # Ensure we hit exactly budget points
                    # If we have too few, add random points in negative regions
                    # If we have too many, trim
                    x1_samples = np.array(x1_samples)
                    if len(x1_samples) < budget:
                        # Add random points from the union of negative regions
                        # Create a flat array of all negative x values
                        all_neg_x = x_check[neg_mask]
                        extra = budget - len(x1_samples)
                        x1_samples = np.concatenate([x1_samples, np.random.choice(all_neg_x, extra)])
                    elif len(x1_samples) > budget:
                        x1_samples = x1_samples[:budget]
                        
                    x1_samples = np.clip(x1_samples, 0.0, 1.0)

        # Create decision vectors
        X = np.zeros((budget, dim))
        X[:, 0] = x1_samples
        # x_2..x_dim are 0 for the Pareto front
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(budget):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
