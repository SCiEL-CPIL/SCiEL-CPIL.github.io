import numpy as np


class Optimizer:
    """Multi-objective optimizer using Adaptive Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f, archive_f):
        """Check if f is dominated by any point in archive_f."""
        if not archive_f:
            return False
        for af in archive_f:
            if np.all(af <= f) and np.any(af < f):
                return True
        return False

    def _update_archive(self, archive_x, archive_f, new_x, new_f):
        """Add new point to archive if non-dominated, and prune archive."""
        if self._is_dominated(new_f, archive_f):
            return archive_x, archive_f, False
        
        new_archive_x = [new_x]
        new_archive_f = [new_f]
        
        for i in range(len(archive_f)):
            af = archive_f[i]
            ax = archive_x[i]
            if np.all(new_f <= af) and np.any(new_f < af):
                continue
            new_archive_x.append(ax)
            new_archive_f.append(af)
            
        max_size = 50
        if len(new_archive_f) > max_size:
            scores = np.sum(new_archive_f, axis=1)
            indices = np.argsort(scores)[:max_size]
            new_archive_x = [new_archive_x[i] for i in indices]
            new_archive_f = [new_archive_f[i] for i in indices]

        return new_archive_x, new_archive_f, True

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        archive_x = []
        archive_f = []
        
        # Phase 1: Random Initialization
        init_size = 50
        for _ in range(init_size):
            if used >= self.budget:
                break
            x = rng.uniform(lo, hi)
            f = problem(x)
            used += 1
            archive_x, archive_f, _ = self._update_archive(archive_x, archive_f, x, f)
        
        if used >= self.budget:
            return np.array(archive_x) if archive_x else np.zeros((1, self.dim))

        # Phase 2: Adaptive Local Search
        step_size = 0.5
        min_step = 0.005
        max_step = 1.0
        stuck_count = 0
        
        while used < self.budget:
            if not archive_x:
                break
                
            if stuck_count >= 10:
                # Inject random point to escape local optima
                x = rng.uniform(lo, hi)
                f = problem(x)
                used += 1
                archive_x, archive_f, accepted = self._update_archive(archive_x, archive_f, x, f)
                if accepted:
                    stuck_count = 0
                step_size = 0.5
                continue

            idx = rng.integers(len(archive_x))
            parent_x = np.array(archive_x[idx])
            
            noise = rng.normal(0, step_size, self.dim)
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            f_child = problem(child_x)
            used += 1
            
            archive_x, archive_f, accepted = self._update_archive(archive_x, archive_f, child_x, f_child)
            
            if accepted:
                step_size = min(step_size * 1.1, max_step)
                stuck_count = 0
            else:
                step_size = max(step_size * 0.9, min_step)
                stuck_count += 1

        if not archive_x:
            return np.zeros((1, self.dim))
            
        return np.array(archive_x)
