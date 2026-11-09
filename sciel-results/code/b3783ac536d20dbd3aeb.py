import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 20
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        
        archive_X = X.copy()
        archive_F = F.copy()
        
        used = N
        
        # DE Parameters
        F_mut = 0.8
        CR = 0.9
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select target
            idx_target = np.random.randint(0, len(archive_X))
            x_target = archive_X[idx_target]
            
            # Select 3 parents for DE
            # Ensure distinct indices if possible
            parents_idx = np.random.choice(len(archive_X), 3, replace=False)
            x_r1 = archive_X[parents_idx[0]]
            x_r2 = archive_X[parents_idx[1]]
            x_r3 = archive_X[parents_idx[2]]
            
            # Mutation
            v = x_r1 + F_mut * (x_r2 - x_r3)
            
            # Crossover
            j_rand = np.random.randint(0, self.dim)
            trial = np.copy(x_target)
            for j in range(self.dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Clip
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Add to archive
            archive_X = np.vstack([archive_X, trial])
            archive_F = np.vstack([archive_F, f_trial])
            
            # Prune Archive to size N
            # 1. Find non-dominated points
            # Dominance: A dominates B if A <= B and A != B
            n_points = len(archive_F)
            dominated = np.zeros(n_points, dtype=bool)
            
            # Efficient dominance check for small N
            for i in range(n_points):
                if dominated[i]: continue
                for j in range(n_points):
                    if i == j or dominated[j]: continue
                    # Check if i dominates j
                    if np.all(archive_F[i] <= archive_F[j]) and np.any(archive_F[i] < archive_F[j]):
                        dominated[j] = True
                    # Check if j dominates i
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        dominated[i] = True
                        break
                        
            # Get non-dominated indices
            nd_indices = np.where(~dominated)[0]
            
            if len(nd_indices) > N:
                # Keep top N by crowding distance
                # Calculate crowding distance for nd_indices
                nd_F = archive_F[nd_indices]
                # Normalize? Not strictly necessary for relative comparison if bounds are similar, 
                # but WFG objectives might have different scales. 
                # Let's just use raw values for crowding distance calculation.
                
                m = len(nd_indices)
                dist = np.zeros(m)
                
                # Sort by each objective
                for k in range(self.n_obj):
                    order = np.argsort(nd_F[:, k])
                    dist[order[0]] = np.inf
                    dist[order[-1]] = np.inf
                    if m > 2:
                        range_k = nd_F[order[-1], k] - nd_F[order[0], k]
                        if range_k > 1e-10:
                            for i in range(1, m-1):
                                dist[order[i]] += (nd_F[order[i+1], k] - nd_F[order[i-1], k]) / range_k
                        else:
                            # If range is 0, all points are same in this obj, 
                            # usually assign 0 or large value? Standard NSGA-II assigns 0 if range is 0? 
                            # Actually, if range is 0, they are identical in this dim.
                            # Let's just leave as is (0) for this dim contribution.
                            pass

                # Select top N by dist
                # Sort indices by dist descending
                sorted_nd = nd_indices[np.argsort(dist)[::-1]]
                keep_indices = sorted_nd[:N]
            else:
                # If <= N non-dominated, keep all non-dominated.
                # Do we need to fill with dominated? 
                # If we only keep non-dominated, the population size shrinks.
                # DE needs a population to sample from.
                # If len(nd_indices) < N, we should probably keep some dominated points to maintain diversity.
                # Let's keep all non-dominated, and fill remaining slots with dominated points 
                # that have the best crowding distance relative to the non-dominated set? 
                # Or simply: if len(nd_indices) < N, keep all non-dominated and random dominated?
                # Better: Keep all non-dominated. If space remains, pick dominated points 
                # that are "closest" to the non-dominated front (or have good crowding among themselves).
                
                # Simplest robust approach: 
                # If len(nd_indices) < N, we keep all non-dominated. 
                # We also keep dominated points that are not dominated by other dominated points?
                # Actually, let's just keep the non-dominated ones. 
                # If the archive shrinks, DE will sample from a smaller set. 
                # To prevent archive from becoming too small (e.g. 1 point), 
                # we can enforce a minimum size by keeping the best dominated points.
                
                # Let's implement: 
                # 1. Keep all non-dominated.
                # 2. If count < N, calculate crowding distance for the remaining dominated points 
                #    (excluding those dominated by non-dominated? No, all dominated).
                #    Actually, just pick random dominated points to fill up to N?
                #    Or better: Pick dominated points with highest crowding distance 
                #    calculated on the *entire* set?
                
                # Let's try: Keep all non-dominated. 
                # If count < N, add dominated points with highest crowding distance 
                # (calculated on the subset of dominated points) until size N.
                
                dominated_indices = np.where(dominated)[0]
                if len(dominated_indices) > 0 and len(nd_indices) < N:
                    need = N - len(nd_indices)
                    dom_F = archive_F[dominated_indices]
                    m_d = len(dom_F)
                    dist_d = np.zeros(m_d)
                    
                    for k in range(self.n_obj):
                        order = np.argsort(dom_F[:, k])
                        dist_d[order[0]] = np.inf
                        dist_d[order[-1]] = np.inf
                        if m_d > 2:
                            range_k = dom_F[order[-1], k] - dom_F[order[0], k]
                            if range_k > 1e-10:
                                for i in range(1, m_d-1):
                                    dist_d[order[i]] += (dom_F[order[i+1], k] - dom_F[order[i-1], k]) / range_k
                            
                    sorted_dom = dominated_indices[np.argsort(dist_d)[::-1]]
                    keep_dominated = sorted_dom[:need]
                    keep_indices = np.concatenate([nd_indices, keep_dominated])
                else:
                    keep_indices = nd_indices

            # Update archive
            archive_X = archive_X[keep_indices]
            archive_F = archive_F[keep_indices]

        return archive_X
