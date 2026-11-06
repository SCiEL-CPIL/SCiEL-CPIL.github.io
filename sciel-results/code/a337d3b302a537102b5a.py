import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115
    rng = random.Random(12345)
    
    def find_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                common = A[a] & A[b]
                common &= ~((1 << (b + 1)) - 1)
                m = common
                while m:
                    low = m & -m
                    c = low.bit_length() - 1
                    m ^= low
                    above = common & ~((1 << (c + 1)) - 1)
                    m2 = above
                    while m2:
                        low2 = m2 & -m2
                        d = low2.bit_length() - 1
                        m2 ^= low2
                        if (A[d] >> c) & 1:
                            return (a, b, c, d)
        return None

    def find_ind_k(n, A, k):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            if found[0] is not None:
                return
            if len(R) == k:
                found[0] = tuple(R)
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < k:
                return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                u_bit = P & -P
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
                
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    def count_k4(n, A):
        cnt = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                common = A[a] & A[b]
                common &= ~((1 << (b + 1)) - 1)
                m = common
                while m:
                    low = m & -m
                    c = low.bit_length() - 1
                    m ^= low
                    above = common & ~((1 << (c + 1)) - 1)
                    m2 = above
                    while m2:
                        low2 = m2 & -m2
                        d = low2.bit_length() - 1
                        m2 ^= low2
                        if (A[d] >> c) & 1:
                            cnt += 1
        return cnt

    def count_ind_k(n, A, k):
        # Approximate count or just check existence? 
        # For SA, we need a smooth energy landscape.
        # Counting all ind-k sets is hard.
        # We can use the number of ind-(k-1) sets that can be extended?
        # Or just use existence (0 or 1) as a penalty.
        # Let's stick to existence for now, it's binary.
        if find_ind_k(n, A, k):
            return 1
        return 0

    def energy(n, A):
        k4 = count_k4(n, A)
        ind6 = count_ind_k(n, A, 6)
        return k4 + 10 * ind6

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def paley_graph(q):
        squares = {x * x % q for x in range(1, q)}
        n = q
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return n, A

    def sa_optimize(n, A, time_limit):
        """Simulated Annealing to minimize energy."""
        cur_A = list(A)
        cur_e = energy(n, cur_A)
        best_A = list(cur_A)
        best_e = cur_e
        
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        start_t = time.time()
        
        while time.time() < time_limit and T > T_min:
            # Flip random edge
            u = rng.randint(0, n-1)
            v = rng.randint(0, n-1)
            if u == v:
                continue
            u, v = min(u, v), max(u, v)
            
            old_edge = (cur_A[v] >> u) & 1
            if old_edge:
                cur_A[u] &= ~(1 << v)
                cur_A[v] &= ~(1 << u)
            else:
                cur_A[u] |= (1 << v)
                cur_A[v] |= (1 << u)
                
            new_e = energy(n, cur_A)
            delta = new_e - cur_e
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                cur_e = new_e
                if new_e < best_e:
                    best_e = new_e
                    best_A = list(cur_A)
                    if best_e == 0:
                        break
            else:
                # Revert
                if old_edge:
                    cur_A[u] |= (1 << v)
                    cur_A[v] |= (1 << u)
                else:
                    cur_A[u] &= ~(1 << v)
                    cur_A[v] &= ~(1 << u)
                    
            T *= cooling
            
        return best_A, best_e

    # Try to find valid graph for n=35, 34, ...
    for n in range(35, 16, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize with Paley(17) padded with random vertices
        # Or just random graph
        A_init = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    A_init[i] |= (1 << j)
                    A_init[j] |= (1 << i)
                    
        # Run SA
        A_opt, e_opt = sa_optimize(n, A_init, deadline - 5)
        
        if e_opt == 0:
            return {'n': n, 'edges': to_edges(n, A_opt)}
            
    # Fallback: Return best found so far or Paley(17)
    # We should track the best valid graph found
    # But if we didn't find any valid for 35..17, we return Paley(17)
    n, A = paley_graph(17)
    return {'n': n, 'edges': to_edges(n, A)}
