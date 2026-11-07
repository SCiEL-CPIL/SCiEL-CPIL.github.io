"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    m = len(sets)
    # Convert to sets for fast subset checking
    set_list = [frozenset(s) for s in sets]
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            si = set_list[i]
            sj = set_list[j]
            if len(si) < len(sj):
                if si.issubset(sj):
                    cnt += 1
            elif len(sj) < len(si):
                if sj.issubset(si):
                    cnt += 1
    return cnt

def _initial_solution():
    """Create a reasonable initial solution using chains."""
    n = P['n']
    m = P['m']
    # Build a solution by taking initial segments of a specific ordering
    # Use colexicographic order or similar to get many comparable pairs
    # Strategy: start with small sets, build up
    all_subsets = []
    for mask in range(1 << n):
        s = tuple(i for i in range(n) if (mask >> i) & 1)
        all_subsets.append(s)
    
    # Sort by (size, then lexicographic) to create chains
    all_subsets.sort(key=lambda x: (len(x), x))
    
    # Take first m sets
    return [list(s) for s in all_subsets[:m]]

def _local_search(sets, time_limit=85.0):
    """Improve the solution via local search (swap operations)."""
    start = time.time()
    n = P['n']
    m = P['m']
    current = [list(s) for s in sets]
    current_score = _count_comparable_pairs(current)
    
    # Precompute all possible subsets as frozensets for quick lookup
    all_frozensets = [frozenset(s) for s in [[i for i in range(n) if (mask >> i) & 1] for mask in range(1 << n)]]
    all_frozensets_set = set(all_frozensets)
    
    # Current set of frozensets
    current_fs = [frozenset(s) for s in current]
    current_fs_set = set(current_fs)
    
    def _score_with_swap(idx, new_set):
        """Compute score change if we replace current[idx] with new_set."""
        # Remove current[idx], add new_set
        old_fs = current_fs[idx]
        new_fs = frozenset(new_set)
        
        if new_fs in current_fs_set and new_fs != old_fs:
            return None  # Duplicate, invalid
        
        # Count comparable pairs involving old_fs
        old_comparable = 0
        for i in range(m):
            if i == idx:
                continue
            other = current_fs[i]
            if len(old_fs) < len(other):
                if old_fs.issubset(other):
                    old_comparable += 1
            elif len(other) < len(old_fs):
                if other.issubset(old_fs):
                    old_comparable += 1
        
        # Count comparable pairs involving new_fs
        new_comparable = 0
        for i in range(m):
            if i == idx:
                continue
            other = current_fs[i]
            if len(new_fs) < len(other):
                if new_fs.issubset(other):
                    new_comparable += 1
            elif len(other) < len(new_fs):
                if other.issubset(new_fs):
                    new_comparable += 1
        
        return new_comparable - old_comparable
    
    # Try swaps
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for idx in range(m):
            if (time.time() - start) > time_limit:
                break
            old_fs = current_fs[idx]
            # Try replacing with each other subset
            for new_s in all_frozensets:
                if new_s == old_fs:
                    continue
                if new_s in current_fs_set:
                    continue
                
                delta = _score_with_swap(idx, new_s)
                if delta is not None and delta > 0:
                    # Accept the swap
                    current_fs_set.remove(old_fs)
                    current_fs[idx] = new_s
                    current[idx] = list(new_s)
                    current_fs_set.add(new_s)
                    current_score += delta
                    improved = True
                    break  # Restart from beginning after improvement
    
    return current, current_score

def construct():
    start = time.time()
    deadline = 85.0  # Safety margin
    
    # Initial solution
    sol = _initial_solution()
    score = _count_comparable_pairs(sol)
    
    # Local search
    sol, score = _local_search(sol, time_limit=deadline - (time.time() - start))
    
    # Ensure types are correct: lists of Python ints
    result = []
    for s in sol:
        result.append([int(x) for x in s])
    
    return result
# EVOLVE-BLOCK-END
