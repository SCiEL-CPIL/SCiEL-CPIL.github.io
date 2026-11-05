"""
Max K(2,2,2)-free triple system on 16 points via optimized greedy + local search.
Uses bitmask-based conflict checking for speed, enabling more search iterations.
"""
import time
import random
import itertools

def construct():
    deadline = time.time() + 112  # safety margin before 150s limit
    random.seed(42)
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    n_triples = len(all_triples)  # 560
    
    # Map each triple to an index for fast lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # For each triple index, precompute the list of "blocking configurations"
    # Each config is a tuple of 7 other triple indices that together with this
    # triple form a K(2,2,2)
    triple_configs = []
    for t in all_triples:
        x, y, z = t
        others = [i for i in range(N) if i not in t]
        configs = []
        for p1 in others:
            for p2 in others:
                if p2 == p1:
                    continue
                for p3 in others:
                    if p3 == p1 or p3 == p2:
                        continue
                    other_triples = [
                        (x, y, p3),
                        (x, p2, z),
                        (x, p2, p3),
                        (p1, y, z),
                        (p1, y, p3),
                        (p1, p2, z),
                        (p1, p2, p3),
                    ]
                    other_indices = []
                    valid = True
                    for ot in other_triples:
                        ot_sorted = tuple(sorted(ot))
                        if ot_sorted in triple_to_idx:
                            other_indices.append(triple_to_idx[ot_sorted])
                        else:
                            valid = False
                            break
                    if valid:
                        configs.append(tuple(other_indices))
        triple_configs.append(configs)
    
    def check_violation_fast(triple_idx, present_set):
        """Check if adding triple at index triple_idx would create K(2,2,2)."""
        if triple_idx in present_set:
            return False
        for config in triple_configs[triple_idx]:
            if all(idx in present_set for idx in config):
                return True
        return False
    
    def greedy_fill(current, max_passes=30):
        """Repeatedly try to add any available triple."""
        for _ in range(max_passes):
            if time.time() > deadline - 8:
                break
            added_any = False
            for t_idx in range(n_triples):
                if t_idx in current:
                    continue
                if not check_violation_fast(t_idx, current):
                    current.add(t_idx)
                    added_any = True
            if not added_any:
                break
        return current
    
    def local_search(current, max_iters=None):
        """1-out, k-in local search with multiple k values."""
        if max_iters is None:
            max_iters = 10000
        iters = 0
        improved = True
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            if len(current) == 0:
                break
            
            # Try removing a random triple
            t_remove = random.choice(tuple(current))
            trial = set(current)
            trial.remove(t_remove)
            
            # Try adding up to 3 new triples
            candidates = [i for i in range(n_triples) if i not in trial and i != t_remove]
            random.shuffle(candidates)
            
            added = 0
            for t_idx in candidates[:200]:
                if time.time() > deadline - 5:
                    break
                if t_idx in trial:
                    continue
                if not check_violation_fast(t_idx, trial):
                    trial.add(t_idx)
                    added += 1
                    if added >= 3:
                        break
            
            if len(trial) > len(current):
                current = trial
                improved = True
        
        return current
    
    def init_star():
        """Create initial triple set: all triples through point 0 (105 triples)."""
        s = set()
        for a in range(1, N):
            for b in range(a+1, N):
                t = (0, a, b)
                s.add(triple_to_idx[t])
        return s
    
    def init_random(p=0.15):
        """Random initialization with probability p of including each triple."""
        s = set()
        for i in range(n_triples):
            if random.random() < p:
                s.add(i)
        # Remove violations
        changed = True
        while changed:
            changed = False
            for t_idx in list(s):
                # Check if this triple creates a violation with others in s
                for config in triple_configs[t_idx]:
                    if all(idx in s for idx in config):
                        s.discard(t_idx)
                        changed = True
                        break
        return s
    
    best = set()
    best_size = 0
    
    # Multiple restarts with different initializations
    num_restarts = 0
    while time.time() < deadline - 15:
        num_restarts += 1
        
        if num_restarts == 1:
            current = init_star()
        elif num_restarts % 3 == 0:
            current = init_random(p=random.uniform(0.1, 0.25))
        else:
            # Perturb best solution
            if best_size > 0:
                current = set(best)
                # Remove a random fraction
                num_remove = max(1, int(len(current) * random.uniform(0.05, 0.15)))
                to_remove = random.sample(list(current), min(num_remove, len(current)))
                for t in to_remove:
                    current.discard(t)
            else:
                current = init_star()
        
        # Greedy fill
        current = greedy_fill(current)
        
        # Local search
        current = local_search(current)
        
        # Final greedy
        current = greedy_fill(current, max_passes=10)
        
        if len(current) > best_size:
            best = set(current)
            best_size = len(best)
        
        # Check if we're running low on time
        if time.time() > deadline - 20:
            break
    
    # Final polish: one more local search on the best
    if time.time() < deadline - 3:
        polished = local_search(set(best), max_iters=5000)
        polished = greedy_fill(polished, max_passes=10)
        if len(polished) > best_size:
            best = polished
            best_size = len(best)
    
    # Convert to list of lists
    result = [list(all_triples[i]) for i in sorted(best)]
    return result
