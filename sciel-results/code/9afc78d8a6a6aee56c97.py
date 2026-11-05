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
        # We need at least 9 elements.
        # We will build the set incrementally.
        # S = [0, s1, s2, ...]
        
        best_solution = None
        
        # Heuristic ordering:
        # We want to cover 1..30.
        # Start with 0.
        # Next, we must cover 1. So 1 must be in S.
        # S = [0, 1]
        # Diffs so far: {1}
        # Need: {2..30}
        
        S = [0, 1]
        used_diffs = {1}
        need_diffs = set(range(2, 31))
        
        # Precompute for each d in 1..D, the pairs (a, a+d)
        # This helps in checking feasibility.
        
        def dfs(current_S, used_diffs, need_diffs):
            nonlocal best_solution
            if best_solution is not None:
                return
            if elapsed() > time_budget:
                return
            
            if not need_diffs:
                best_solution = list(current_S)
                return
                
            # Pruning:
            # 1. Size bound: We have k elements. Max possible new diffs if we add up to D-current_max is limited.
            #    But simpler: if we run out of space, fail.
            # 2. Capacity:
            #    Current size k = len(current_S).
            #    Max size allowed by D is D+1, but Sidon sets are sparse.
            #    Max number of elements in a Sidon set in [0, D] is approx sqrt(D).
            #    For D=44, max size is about 7? No, 9 is possible.
            #    Let's just check if we can cover the remaining needs.
            
            k = len(current_S)
            # If we add m more elements, we get m*k + m*(m-1)/2 new diffs.
            # We need to cover len(need_diffs) diffs.
            # Max m we can add is limited by D - current_S[-1].
            max_add = D - current_S[-1]
            # Find min m such that m*k + m*(m-1)/2 >= len(need_diffs)
            # If m > max_add, fail.
            m = 1
            cap = 0
            while m <= max_add and cap < len(need_diffs):
                cap = m * k + m * (m - 1) // 2
                if cap < len(need_diffs):
                    m += 1
            if m > max_add:
                return
                
            # Generate candidates for the next element x
            # x must be > current_S[-1] and <= D
            # x must not create duplicate diffs with current_S
            # x should cover as many need_diffs as possible
            
            candidates = []
            last = current_S[-1]
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
            
            if not candidates:
                return
            
            # Sort by new_cover desc, then x asc
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for new_cover, x in candidates:
                # Optional: If new_cover == 0, we are just padding. 
                # We should only do this if we are close to the limit.
                # But for minimal D, we probably need every element to contribute.
                
                # Add x
                current_S.append(x)
                new_used = set(used_diffs)
                new_need = set(need_diffs)
                for s in current_S[:-1]:
                    d = x - s
                    new_used.add(d)
                    if d in new_need:
                        new_need.remove(d)
                
                dfs(current_S, new_used, new_need)
                
                # Undo
                current_S.pop()
                
                if best_solution is not None:
                    return

        dfs(S, used_diffs, need_diffs)
        return best_solution

    # Search for minimal D
    # Lower bound:
    # Min diameter for Sidon set of size 9 is 44.
    # We check D from 44 upwards.
    
    best_set = None
    for D in range(44, 60):
        if elapsed() > deadline - 10.0:
            break
        # Allocate time budget based on remaining time
        budget = min(10.0, deadline - elapsed() - 5.0)
        if budget <= 0:
            break
        sol = solve_for_D(D, budget)
        if sol:
            best_set = sol
            break # Found minimal D
            
    if best_set is None:
        # Fallback: Greedy construction
        best_set = [0]
        used = set()
        need = set(range(1, 31))
        while need and elapsed() < deadline - 2.0:
            best_x = None
            best_c = -1
            # Search up to 100
            for x in range(max(best_set) + 1, max(best_set) + 100):
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
                # Try wider
                for x in range(max(best_set) + 100, max(best_set) + 500):
                    ok = True
                    for s in best_set:
                        if x - s in used:
                            ok = False
                            break
                    if ok:
                        best_x = x
                        break
            if best_x is None:
                break
            best_set.append(best_x)
            for s in best_set[:-1]:
                used.add(best_x - s)
            need -= used

    # Verify
    if not is_sidon(best_set):
        # Last resort known set?
        # Just return the best effort
        pass

    return best_set

if __name__ == "__main__":
    print(construct())
