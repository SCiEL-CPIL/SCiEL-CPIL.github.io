import sys
sys.setrecursionlimit(20000)

def construct(n_rows, k):
    # 1. Baseline Greedy
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    baseline_max = max(max(r) for r in baseline_rows)
    
    # Theoretical lower bound
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    # M(M-1)/2 is not the right bound for distinct diffs in {1..M}.
    # The diffs are just distinct integers. So M >= total_diffs_needed.
    min_possible_max = total_diffs_needed
    
    # If the baseline is already at the lower bound, return it
    if baseline_max <= min_possible_max:
        return baseline_rows

    def get_candidates(max_val):
        candidates = []
        
        def backtrack_ruler(current, used_diffs):
            if len(current) == k + 1:
                candidates.append(current[:])
                return
            
            last = current[-1]
            # Optimization: We can limit the search range for x
            # But we must go up to max_val.
            
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in current}
                if not (new_diffs & used_diffs):
                    current.append(x)
                    backtrack_ruler(current, used_diffs | new_diffs)
                    current.pop()
        
        backtrack_ruler([0], set())
        return candidates

    def find_disjoint_subset(candidates, n_needed):
        # Precompute diffs
        cand_info = []
        for c in candidates:
            diffs = frozenset(c[j] - c[i] for i in range(len(c)) for j in range(i+1, len(c)))
            cand_info.append((c, diffs))
        
        # Sort by max value to find small scope solutions first
        cand_info.sort(key=lambda x: max(x[0]))
        
        solution = []
        
        def dfs(idx, used_diffs, selected):
            if len(selected) == n_needed:
                solution.extend(selected)
                return True
            
            # Pruning: not enough items left
            if len(cand_info) - idx < n_needed - len(selected):
                return False
            
            for i in range(idx, len(cand_info)):
                c, diffs = cand_info[i]
                if not (diffs & used_diffs):
                    selected.append(c)
                    if dfs(i + 1, used_diffs | set(diffs), selected):
                        return True
                    selected.pop()
            
            return False
        
        if dfs(0, set(), []):
            return solution
        return None

    # Search for better solution
    # To avoid timeout, we limit the search scope increase
    # If baseline_max is 112, we check 105..111
    
    for M in range(min_possible_max, baseline_max):
        candidates = get_candidates(M)
        if not candidates:
            continue
        
        # If the number of candidates is very large, the DFS might be slow.
        # But for k=5, it's usually manageable.
        
        sol = find_disjoint_subset(candidates, n_rows)
        if sol:
            return sol
            
    return baseline_rows
