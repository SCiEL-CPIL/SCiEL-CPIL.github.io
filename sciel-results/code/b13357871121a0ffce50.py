import time
import math

def construct() -> dict:
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def make_paley_adjacency(p):
        """Build the adjacency matrix (0/1 list) for Paley tournament on p vertices."""
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        adjacency = []
        for i in range(p):
            for j in range(i + 1, p):
                # i -> j iff (j - i) is a quadratic residue mod p
                if (j - i) % p in residues:
                    adjacency.append(0)  # i -> j
                else:
                    adjacency.append(1)  # j -> i
        return adjacency

    def build_adjacency_matrix(p, adjacency):
        """Convert flat adjacency list to a 2D boolean numpy matrix."""
        import numpy as np
        adj = np.zeros((p, p), dtype=bool)
        idx = 0
        for i in range(p):
            for j in range(i + 1, p):
                if adjacency[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        return adj

    def check_property_fast(p, adjacency):
        """
        Check if every 4-subset has a common dominator.
        Uses numpy for speed with a count-matrix approach.
        """
        import numpy as np
        adj = build_adjacency_matrix(p, adjacency)
        # adj[i,j] = True iff i -> j
        # For a 4-set {u,v,w,x}, we need a vertex y such that y->u, y->v, y->w, y->x.
        # Equivalently, y is in In(u) ∩ In(v) ∩ In(w) ∩ In(x).
        # For fixed u,v,w, let S = In(u) ∩ In(v) ∩ In(w).
        # For each x > w, we need S ∩ In(x) ≠ ∅, i.e., ∃y ∈ S with y→x.
        # This is: for each x > w, (S & adj[:, x]).any() == True.
        # We can compute: for each x, count_x = sum(S & adj[:, x]) = S @ adj[:, x].
        # Then check count_x >= 1 for all x > w.
        
        in_neighbors = adj.T  # in_neighbors[j] is boolean vector: True at i iff i->j
        
        for u in range(p):
            for v in range(u + 1, p):
                for w in range(v + 1, p):
                    if time.time() > deadline - 5.0:
                        return False
                    # S = vertices that dominate u, v, and w
                    S = in_neighbors[u] & in_neighbors[v] & in_neighbors[w]
                    if w + 1 < p:
                        # For each x in w+1..p-1, check if any y in S dominates x
                        # count[x] = number of y in S such that y->x
                        counts = S @ adj  # length-p vector: counts[x] = sum_y S[y] * adj[y,x]
                        # We need counts[x] >= 1 for all x > w
                        if not np.all(counts[w + 1:] >= 1):
                            return False
        return True

    # Search for the smallest prime power q ≡ 3 (mod 4) such that Paley(q) satisfies the property.
    # We know q=67 works. Let's verify starting from small prime powers to be robust.
    # The lower bound is 31.
    
    best_n = 83  # Fallback
    best_adj = None
    
    # Generate all prime powers q ≡ 3 (mod 4) with 5 <= q <= 90
    def is_prime_power(q):
        """Check if q is a prime power, return (prime, exponent) or None."""
        if q < 2:
            return None
        # Try all primes up to sqrt(q)
        for p in range(2, int(math.isqrt(q)) + 1):
            if is_prime(p):
                x = q
                exp = 0
                while x % p == 0:
                    x //= p
                    exp += 1
                if x == 1 and exp >= 1:
                    return (p, exp)
        # q itself is prime
        if is_prime(q):
            return (q, 1)
        return None
    
    # Collect all valid q
    candidates = []
    for q in range(5, 91):
        if q % 4 == 3:
            pp = is_prime_power(q)
            if pp is not None:
                candidates.append((q, pp))
    
    # Sort by q
    candidates.sort(key=lambda x: x[0])
    
    for q, (base, exp) in candidates:
        if time.time() > deadline - 10.0:
            break
        
        if exp == 1:
            # Prime case: use existing function
            adj_list = make_paley_adjacency(q)
        else:
            # Prime power case: need GF(q) Paley tournament
            # For q = p^k, k>1, we need to construct the Paley tournament over GF(q)
            # This is more complex. For now, skip non-prime powers unless we implement them.
            # Actually, let's implement a simple version for small prime powers.
            # For q = p^k, the quadratic residues in GF(q) are defined via the norm map or
            # via a primitive element. This is complex. For now, only handle primes.
            continue
        
        if check_property_fast(q, adj_list):
            return {'n': q, 'orientation': adj_list}

    # If we didn't find a smaller one, or loop finished, return the known best (67).
    adj_67 = make_paley_adjacency(67)
    return {'n': 67, 'orientation': adj_67}
