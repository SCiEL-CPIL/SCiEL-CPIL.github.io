"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0  # stay safely under 30 s
    nodes_visited = 0
    max_nodes = 50000  # Limit DFS complexity

    def sums_set(S):
        T = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            T.add(2 * a)
            for b in L[i + 1:]:
                T.add(a + b)
        return T

    def is_sidon(S):
        seen = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for b in L[i + 1:]:
                s = a + b
                if s in seen:
                    return False
                seen.add(s)
        return True

    def get_addable(S, T):
        res = []
        Sset = set(S)
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if 2 * x in T:
                continue
            ok = True
            for s in S:
                if x + s in T:
                    ok = False
                    break
            if ok:
                res.append(x)
        return res

    def calc_blocking(S, T, x, addable):
        # How many elements in 'addable' are blocked by adding x?
        new_sums = {2 * x}
        for s in S:
            new_sums.add(x + s)
        blocked = 0
        Sset = set(S) | {x}
        T2 = T | new_sums
        for y in addable:
            if y == x:
                continue
            if 2 * y in T2:
                blocked += 1
                continue
            ok = True
            for s in Sset:
                if y + s in T2:
                    ok = False
                    break
            if not ok:
                blocked += 1
        return blocked

    def dfs(S, T, addable, limit):
        nonlocal nodes_visited
        if time.time() > deadline or nodes_visited > max_nodes:
            return None
        
        nodes_visited += 1
        
        if not addable:
            if len(S) <= limit:
                return S[:]
            else:
                return None
        
        if len(S) >= limit:
            return None

        # Heuristic: Sort candidates by blocking power (descending)
        # We want to try candidates that block the most first, to reach maximality faster
        candidates = []
        for x in addable:
            block_count = calc_blocking(S, T, x, addable)
            candidates.append((block_count, x))
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        for _, x in candidates:
            # Add x
            new_S = S + [x]
            new_sums = {2 * x}
            for s in S:
                new_sums.add(x + s)
            new_T = T | new_sums
            
            # Update addable
            # Optimization: we can just call get_addable, but it's O(N^2).
            # Better to filter existing addable list.
            new_addable = []
            Sset = set(new_S)
            for y in addable:
                if y == x:
                    continue
                if 2 * y in new_T:
                    continue
                ok = True
                for s in new_S:
                    if y + s in new_T:
                        ok = False
                        break
                if ok:
                    new_addable.append(y)
            
            res = dfs(new_S, new_T, new_addable, limit)
            if res is not None:
                return res
        
        return None

    # 1. Greedy Construction
    S_greedy = []
    T_greedy = set()
    addable = get_addable(S_greedy, T_greedy)
    
    while addable and time.time() < deadline:
        best_x = None
        best_block = -1
        # Find x that blocks the most
        for x in addable:
            block_count = calc_blocking(S_greedy, T_greedy, x, addable)
            if block_count > best_block:
                best_block = block_count
                best_x = x
        
        if best_x is None:
            break
        
        S_greedy.append(best_x)
        new_sums = {2 * best_x}
        for s in S_greedy[:-1]:
            new_sums.add(best_x + s)
        T_greedy |= new_sums
        addable = get_addable(S_greedy, T_greedy) # Recompute to be safe, or filter
        
    best_set = S_greedy[:]
    best_size = len(best_set)
    
    # 2. Try to improve with DFS
    # We want to find a set of size < best_size
    # Iterate down from best_size - 1
    for limit in range(best_size - 1, 5, -1):
        if time.time() > deadline:
            break
        nodes_visited = 0
        # Reset max_nodes for each limit? Or keep global?
        # Keeping global prevents total timeout.
        res = dfs([], set(), get_addable([], set()), limit)
        if res is not None:
            best_set = res
            best_size = len(res)
            # If we found one, we can try to go even smaller?
            # The loop continues to limit-1, so yes.
            # But we should probably restart DFS with fresh state?
            # The dfs function is pure-ish (uses nonlocal counter).
            # We can just continue the loop.
            
    # Final verification
    if not is_sidon(best_set):
        # Fallback to simple greedy if something went wrong
        S_fallback = []
        T_fallback = set()
        for x in range(1, N + 1):
            added = [x + a for a in S_fallback] + [2 * x]
            if len(set(added)) == len(added) and not T_fallback.intersection(added):
                S_fallback.append(x)
                T_fallback.update(added)
        best_set = S_fallback
        
    return best_set
# EVOLVE-BLOCK-END
