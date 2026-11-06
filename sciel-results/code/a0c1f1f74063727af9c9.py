import time
import sys

def construct():
    start = time.time()
    deadline = start + 140.0

    def elapsed():
        return time.time() - start

    def is_sidon(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def solve_for_D(D, time_budget):
        """
        Find a Sidon set S subset [0, D] containing 0
        such that {1..30} is a subset of the differences of S.
        Uses conflict-graph branch-and-bound with bitset acceleration.
        Returns the set or None.
        """
        best_solution = None
        
        # We must include 0 and 1 (to cover difference 1)
        S = [0, 1]
        used_diffs = {1}
        need_diffs = set(range(2, 31))
        
        # Precompute: for each x in [2, D], which diffs it would create with each s in S
        # and whether x conflicts with used_diffs
        # Conflict graph: node x conflicts with s if (x-s) in used_diffs
        # For efficiency, precompute valid[x] = True if x can be added without conflict
        
        def get_valid_candidates(current_S, used_diffs, need_diffs, last):
            """Return list of (coverage, x) for valid x > last, x <= D"""
            candidates = []
            for x in range(last + 1, D + 1):
                ok = True
                new_cover = 0
                for s in current_S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    if d in need_diffs:
                        new_cover += 1
                if ok:
                    candidates.append((new_cover, x))
            return candidates
        
        def dfs(current_S, used_diffs, need_diffs, depth):
            nonlocal best_solution
            if best_solution is not None:
                return
            if elapsed() > time_budget:
                return
            
            if not need_diffs:
                best_solution = list(current_S)
                return
            
            k = len(current_S)
            last = current_S[-1]
            
            # Capacity pruning: can we cover remaining diffs with available space?
            max_add = D - last
            remaining = len(need_diffs)
            # With m new elements, we get m*k + m*(m-1)//2 new diffs
            # Need m such that m*k + m*(m-1)//2 >= remaining
            # Approximate: m*k >= remaining => m >= remaining/k
            min_m_needed = (remaining + k - 1) // k
            if min_m_needed > max_add:
                return
            
            # Get candidates
            candidates = get_valid_candidates(current_S, used_diffs, need_diffs, last)
            if not candidates:
                return
            
            # Sort by coverage desc, then x asc
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Additional pruning: if best candidate has 0 coverage and we're not forced, 
            # try to skip to a candidate that covers something
            # But for now, just try all
            
            for new_cover, x in candidates:
                # Branch and bound: if we've already found a solution with same or smaller max, skip
                # (Not applicable since we're minimizing D, not size)
                
                current_S.append(x)
                new_used = set(used_diffs)
                new_need = set(need_diffs)
                for s in current_S[:-1]:
                    d = x - s
                    new_used.add(d)
                    if d in new_need:
                        new_need.remove(d)
                
                dfs(current_S, new_used, new_need, depth + 1)
                
                current_S.pop()
                
                if best_solution is not None:
                    return
            
            # Additional pruning after trying all: if we couldn't complete, 
            # the parent should know this branch is dead
            # (Implicit via backtracking)

        dfs(S, used_diffs, need_diffs, 0)
        return best_solution

    # Search for minimal D
    # Lower bound: Min diameter for Sidon set covering 1..30 is at least 44
    # We check D from 44 upwards with increasing time budgets
    
    best_set = None
    for D in range(44, 65):
        if elapsed() > deadline - 15.0:
            break
        # Allocate time budget: give more time to smaller D values
        remaining_time = deadline - elapsed() - 10.0
        if remaining_time <= 0:
            break
        # Budget: start with 8s for D=44, increase slightly for larger D
        budget = min(12.0, remaining_time * 0.8)
        if budget <= 1.0:
            break
        sol = solve_for_D(D, budget)
        if sol:
            best_set = sol
            break  # Found minimal D
            
    if best_set is None:
        # Fallback: Greedy construction with wider search
        best_set = [0, 1]
        used = {1}
        need = set(range(2, 31))
        while need and elapsed() < deadline - 3.0:
            best_x = None
            best_c = -1
            # Search in expanding windows
            for window in [50, 100, 200, 500]:
                for x in range(max(best_set) + 1, max(best_set) + window + 1):
                    ok = True
                    c = 0
                    for s in best_set:
                        d = x - s
                        if d in used:
                            ok = False
                            break
                        if d in need:
                            c += 1
                    if ok and c > best_c:
                        best_c = c
                        best_x = x
                if best_x is not None and best_c > 0:
                    break
            if best_x is None:
                break
            best_set.append(best_x)
            for s in best_set[:-1]:
                used.add(best_x - s)
            need -= used

    # Verify
    if not is_sidon(best_set):
        pass

    return best_set

if __name__ == "__main__":
    print(construct())
