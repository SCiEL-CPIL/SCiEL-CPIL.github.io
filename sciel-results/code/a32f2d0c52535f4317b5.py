import time
import random

def construct():
    deadline = time.time() + 140  # 150s kill, 10s safety margin
    rng = random.Random(12345)

    # ---------- Fast helpers ----------
    def k4_count(n, A):
        cnt = 0
        for a in range(n):
            na = A[a]
            for b in range(a + 1, n):
                if not (na >> b) & 1:
                    continue
                common = A[b] & na
                for c in range(b + 1, n):
                    if not (common >> c) & 1:
                        continue
                    common2 = A[c] & common
                    # count bits above c
                    m = common2 >> (c + 1)
                    while m:
                        m &= m - 1
                        cnt += 1
        return cnt

    def find_k4(n, A):
        for a in range(n):
            na = A[a]
            for b in range(a + 1, n):
                if not (na >> b) & 1:
                    continue
                common = A[b] & na
                for c in range(b + 1, n):
                    if not (common >> c) & 1:
                        continue
                    common2 = A[c] & common
                    m = common2 >> (c + 1)
                    if m:
                        d_bit = m & -m
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None

    def has_ind6(n, A):
        # Greedy + random restarts: try to build independent set of size 6
        # Complement graph: edge in complement = non-edge in A
        # We do a quick randomized greedy: repeatedly pick a vertex with fewest
        # neighbours in the remaining candidate set (in the original graph),
        # which means fewest non-neighbours in complement... 
        # Actually for independent set in G, we need vertices with no edges between them.
        # Greedy: pick vertex, remove all its neighbours. Repeat.
        # Randomized: shuffle order, try multiple times.
        
        for trial in range(20):
            remaining = set(range(n))
            chosen = []
            # Randomize by shuffling a copy
            verts = list(remaining)
            rng.shuffle(verts)
            for v in verts:
                if v not in remaining:
                    continue
                chosen.append(v)
                # Remove v and all its neighbours from remaining
                remaining.discard(v)
                for u in remaining:
                    if (A[v] >> u) & 1:
                        remaining.discard(u)
                if len(chosen) >= 6:
                    return True
        # Also try a deterministic greedy: always pick vertex with min degree in remaining
        for trial in range(10):
            remaining = set(range(n))
            chosen = []
            while remaining and len(chosen) < 6:
                # Pick vertex with minimum degree within remaining
                best_v = None
                best_deg = n + 1
                for v in remaining:
                    deg = 0
                    for u in remaining:
                        if u != v and (A[v] >> u) & 1:
                            deg += 1
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                if best_v is None:
                    break
                chosen.append(best_v)
                remaining.discard(best_v)
                for u in list(remaining):
                    if (A[best_v] >> u) & 1:
                        remaining.discard(u)
            if len(chosen) >= 6:
                return True
        return False

    def cost(n, A):
        """Return (k4_count, has_ind6) as a combined penalty."""
        k4c = k4_count(n, A)
        if k4c > 0:
            return k4c
        if has_ind6(n, A):
            return 1
        return 0

    # ---------- Paley 17 seed ----------
    def paley_17():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        return A

    # ---------- Simulated Annealing on edge flips for fixed n ----------
    def anneal(n, A, start_temp, end_temp, max_iter):
        """SA over single-edge flips. Returns best A found."""
        A = list(A)
        cur_cost = cost(n, A)
        best_A = list(A)
        best_cost = cur_cost
        temp = start_temp
        it = 0
        while it < max_iter and time.time() < deadline - 5:
            temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
            # Pick random edge to flip
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                it += 1
                continue
            # Flip edge (i,j)
            if (A[i] >> j) & 1:
                A[i] &= ~(1 << j)
                A[j] &= ~(1 << i)
            else:
                A[i] |= (1 << j)
                A[j] |= (1 << i)
            
            new_cost = cost(n, A)
            delta = new_cost - cur_cost
            
            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / (temp + 1e-9))):
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = list(A)
            else:
                # Revert
                if (A[i] >> j) & 1:
                    A[i] &= ~(1 << j)
                    A[j] &= ~(1 << i)
                else:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
            
            it += 1
        return best_A, best_cost

    # ---------- Main: progressive n with SA ----------
    best_n = 17
    best_A = paley_17()
    
    # Verify seed
    assert cost(best_n, best_A) == 0, f"Paley 17 should be valid, cost={cost(best_n, best_A)}"
    
    # Try to increase n from 18 up to 40
    for target_n in range(18, 41):
        if time.time() > deadline - 15:
            break
        
        # Initialize: take current best_A (on best_n vertices) and add new vertex
        # with random connections
        init_A = list(best_A)
        init_A.append(0)
        # Add random edges from new vertex
        density = 0.3 + rng.random() * 0.3
        for i in range(best_n):
            if rng.random() < density:
                init_A[i] |= (1 << (target_n - 1))
                init_A[target_n - 1] |= (1 << i)
        
        # Also try a few different initializations
        best_init_A = init_A
        best_init_cost = cost(target_n, init_A)
        
        for init_trial in range(5):
            if time.time() > deadline - 15:
                break
            trial_A = list(best_A)
            trial_A.append(0)
            density = 0.2 + rng.random() * 0.5
            for i in range(best_n):
                if rng.random() < density:
                    trial_A[i] |= (1 << (target_n - 1))
                    trial_A[target_n - 1] |= (1 << i)
            tc = cost(target_n, trial_A)
            if tc < best_init_cost:
                best_init_cost = tc
                best_init_A = trial_A
        
        # Run SA
        max_iter = 3000
        result_A, result_cost = anneal(target_n, best_init_A, 10.0, 0.01, max_iter)
        
        if result_cost == 0:
            # Success! Update best
            best_n = target_n
            best_A = result_A
        else:
            # Failed to find valid graph at this n; try a few more times
            found = False
            for retry in range(3):
                if time.time() > deadline - 15:
                    break
                retry_A = list(best_A)
                retry_A.append(0)
                density = rng.random() * 0.6
                for i in range(best_n):
                    if rng.random() < density:
                        retry_A[i] |= (1 << (target_n - 1))
                        retry_A[target_n - 1] |= (1 << i)
                rA, rc = anneal(target_n, retry_A, 10.0, 0.01, max_iter)
                if rc == 0:
                    best_n = target_n
                    best_A = rA
                    found = True
                    break
            if not found:
                # Give up on this n, try next
                continue

    # Final verification
    final_cost = cost(best_n, best_A)
    if final_cost != 0:
        # Fallback: return the 17-vertex Paley graph
        best_n = 17
        best_A = paley_17()

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_A[j] >> i) & 1:
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}")
