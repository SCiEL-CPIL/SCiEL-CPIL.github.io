import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    def check_tournament(n, adj_matrix):
        """Check if every 4-set has a common dominator.
        adj_matrix[i][j] = 1 if i -> j, else 0.
        Returns True if valid."""
        # in_neighbors[u] is bitmask of vertices that point to u
        in_neighbors = np.zeros(n, dtype=np.uint64)
        for u in range(n):
            mask = 0
            for v in range(n):
                if v == u:
                    continue
                if adj_matrix[v][u]:
                    mask |= (1 << v)
            in_neighbors[u] = mask
        
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(i + 1, n):
                if not valid:
                    break
                for k in range(j + 1, n):
                    if not valid:
                        break
                    inter_ijk = in_neighbors[i] & in_neighbors[j] & in_neighbors[k]
                    for l in range(k + 1, n):
                        if (inter_ijk & in_neighbors[l]) == 0:
                            valid = False
                            break
        return valid

    def is_prime(p):
        if p < 2:
            return False
        if p < 4:
            return True
        if p % 2 == 0 or p % 3 == 0:
            return False
        i = 5
        while i * i <= p:
            if p % i == 0 or p % (i + 2) == 0:
                return False
            i += 6
        return True

    def get_prime_powers_q3mod4(max_n=90):
        """Get all prime powers q <= max_n with q ≡ 3 (mod 4)."""
        results = set()
        # Check all numbers 3..max_n for being prime powers
        for q in range(3, max_n + 1):
            if q % 4 != 3:
                continue
            # Check if q is a prime power
            found = False
            for p in range(2, int(q ** 0.5) + 2):
                if not is_prime(p):
                    continue
                temp = q
                while temp % p == 0:
                    temp //= p
                if temp == 1:
                    found = True
                    break
            if found:
                results.add(q)
        return sorted(results)

    def make_paley(q):
        """Create adjacency matrix for Paley tournament on q vertices.
        q must be a prime power with q ≡ 3 (mod 4)."""
        # Compute quadratic residues mod q
        squares = set()
        for x in range(q):
            squares.add((x * x) % q)
        
        adj = np.zeros((q, q), dtype=np.uint8)
        for u in range(q):
            for v in range(q):
                if u == v:
                    continue
                diff = (v - u) % q
                if diff in squares and diff != 0:
                    adj[u][v] = 1
        return adj

    # Get all prime powers q ≡ 3 (mod 4) up to 90
    candidates = get_prime_powers_q3mod4(90)
    
    # Also include primes ≡ 3 (mod 4) as a fallback check (they should be in candidates)
    for p in range(3, 91):
        if p % 4 == 3 and is_prime(p):
            if p not in candidates:
                candidates.append(p)
    candidates = sorted(set(candidates))

    # Try each candidate prime power in increasing order
    for q in candidates:
        if time.time() - start_time > deadline - 5:
            break
        
        adj = make_paley(q)
        
        if check_tournament(q, adj):
            orientation = []
            for i in range(q):
                for j in range(i + 1, q):
                    if adj[i][j]:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': q, 'orientation': orientation}

    # Fallback: return Paley(67) which is known to work
    q = 67
    adj = make_paley(q)
    orientation = []
    for i in range(q):
        for j in range(i + 1, q):
            if adj[i][j]:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': q, 'orientation': orientation}
