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
        Returns the set or None.
        """
        best_solution = None
        
        # Precompute: for each x in [2, D], which diffs it creates with [0,1]
        # and whether it's compatible with the base set {0, 1}
        # A candidate x is compatible if (x-0) and (x-1) are distinct and not in {1}
        
        # Actually, let's precompute for each x in [2, D]:
        # - diffs_with_base = {x, x-1} (diffs with 0 and 1)
        # - compatible_with_base = (x != 1) and (x-1 != 1) and (x != x-1) [always true for x>=2]
        #   So compatible if x >= 2 and x-1 != 1, i.e., x >= 3
        #   Wait: x=2 gives diffs {2, 1}. 1 is already used. So x=2 is incompatible.
        #   x=3 gives diffs {3, 2}. Both new. Compatible.
        
        # Let's just do the DFS but with better candidate generation.
        
        S = [0, 1]
        used_diffs = {1}
        need_diffs = set(range(2, 31))
        
        # Precompute for each x in [2, D]: the set of diffs it would create with any subset
        # This is hard. Instead, let's use a different approach.
        
        # Key insight: We need exactly 7 more elements (total 9) to cover 29 more diffs.
        # With 7 elements added to a set of size 2, we get at most 7*2 + 7*6/2 = 14 + 21 = 35 new diffs.
        # We need 29. So we have some slack.
        
        def dfs(current_S, used_diffs, need_diffs, depth):
            nonlocal best_solution
            if best_solution is not None:
                return
            if elapsed() > time_budget:
                return
            
            if not need_diffs:
                best_solution = list(current_S)
                return
                
            # Capacity check
            k = len(current_S)
            remaining_slots = D - current_S[-1]
            # We need to cover len(need_diffs) more diffs
            # With m more elements, we get m*k + m*(m-1)/2 diffs
            # Find min m needed
            m_needed = 1
            while m_needed <= remaining_slots:
                cap = m_needed * k + m_needed * (m_needed - 1) // 2
                if cap >= len(need_diffs):
                    break
                m_needed += 1
            if m_needed > remaining_slots:
                return
            
            # Generate candidates: x in [current_S[-1]+1, D]
            # x is valid if for all s in current_S, (x-s) not in used_diffs
            # Score by how many need_diffs it covers
            
            last = current_S[-1]
            candidates = []
            for x in range(last + 1, D + 1):
                ok = True
                cover = 0
                for s in current_S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    if d in need_diffs:
                        cover += 1
                if ok:
                    candidates.append((cover, x))
            
            if not candidates:
                return
            
            # Sort by cover desc, then x asc
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates to top N for speed
            max_candidates = 15
            for cover, x in candidates[:max_candidates]:
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

        dfs(S, used_diffs, need_diffs, 0)
        return best_solution

    # Search for minimal D
    best_set = None
    for D in range(44, 60):
        if elapsed() > deadline - 10.0:
            break
        budget = min(15.0, deadline - elapsed() - 5.0)
        if budget <= 0:
            break
        sol = solve_for_D(D, budget)
        if sol:
            best_set = sol
            break

    if best_set is None:
        # Fallback: Greedy construction
        best_set = [0]
        used = set()
        need = set(range(1, 31))
        while need and elapsed() < deadline - 2.0:
            best_x = None
            best_c = -1
            for x in range(max(best_set) + 1, max(best_set) + 200):
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
            if best_x is None:
                break
            best_set.append(best_x)
            for s in best_set[:-1]:
                used.add(best_x - s)
            need -= used

    if not is_sidon(best_set):
        pass

    return best_set

if __name__ == "__main__":
    print(construct())
