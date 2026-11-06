"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin

def construct():
    """
    Returns a valid coloring for n vertices.
    Strategy:
    1. Use a Paley-like construction (quadratic residues) to generate a structured initial coloring.
    2. Use vectorized NumPy operations to quickly repair violations (red K4, blue K5).
    3. If a valid coloring is found for a large n, return it.
    4. Fall back to smaller n if time runs out.
    """
    
    def _triples_to_idx(n):
        triples = list(combinations(range(n), 3))
        m = len(triples)
        idx = {}
        for i, t in enumerate(triples):
            idx[t] = i
        return triples, idx, m

    def _build_quads(n):
        m = len(list(combinations(range(n), 3)))
        def t_idx(a, b, c):
            cnt = 0
            for x in range(a):
                cnt += (n - x - 1) * (n - x - 2) // 2
            for y in range(a + 1, b):
                cnt += (n - y - 1)
            cnt += (c - b - 1)
            return cnt
        quads = []
        for q in combinations(range(n), 4):
            a, b, c, d = q
            idxs = [t_idx(a, b, c), t_idx(a, b, d), t_idx(a, c, d), t_idx(b, c, d)]
            quads.append(idxs)
        return quads

    def _build_pents(n):
        m = len(list(combinations(range(n), 3)))
        def t_idx(a, b, c):
            cnt = 0
            for x in range(a):
                cnt += (n - x - 1) * (n - x - 2) // 2
            for y in range(a + 1, b):
                cnt += (n - y - 1)
            cnt += (c - b - 1)
            return cnt
        pents = []
        for p in combinations(range(n), 5):
            a, b, c, d, e = p
            idxs = []
            for t in combinations(p, 3):
                idxs.append(t_idx(*t))
            pents.append(idxs)
        return pents

    def _is_square_mod_n(x, n):
        if x % n == 0:
            return True
        # Check if x is a quadratic residue mod n
        # For prime n, x^((n-1)/2) == 1
        # For general n, we can just check if there exists y such that y^2 == x mod n
        # But for Paley, n is usually prime or 1 mod 4.
        # Let's assume n is prime for the Paley construction.
        if n == 2: return True
        # Simple check for small n
        for y in range(n):
            if (y * y) % n == x % n:
                return True
        return False

    def _paley_initial(n):
        """
        Generate initial coloring based on Paley tournament structure.
        Color (a, b, c) red (0) if the number of "forward" edges in the induced tournament
        is even, or based on some invariant.
        Let's use: Red if (b-a) is a QR mod n and (c-b) is a QR mod n and (c-a) is a QR mod n?
        Actually, a common construction for R(4,5;3) > 32 uses specific algebraic structures.
        Let's try a simple heuristic:
        Red if (a*b + b*c + c*a) % n == 0?
        Or better: Use the property that in a Paley tournament, the out-degree is (n-1)/2.
        Let's define a tournament T on n vertices where edge (i,j) is i->j if (j-i) is a QR mod n.
        Then color triple (a,b,c) red if the induced subgraph on {a,b,c} has a cyclic orientation (1 cycle).
        In a tournament on 3 vertices, it's either transitive or cyclic.
        Cyclic if exactly one of (a->b, b->c, c->a) or (a->c, c->b, b->a) holds?
        Actually, just check if the number of edges in the cycle is 3.
        """
        # Precompute QR table for speed
        qr_table = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                diff = (j - i) % n
                if diff == 0:
                    qr_table[i, j] = True # Self loop, doesn't matter for distinct
                else:
                    qr_table[i, j] = _is_square_mod_n(diff, n)
        
        triples, idx, m = _triples_to_idx(n)
        colors = np.zeros(m, dtype=np.int8)
        
        for i, (a, b, c) in enumerate(triples):
            # Check orientation
            ab = qr_table[a, b]
            bc = qr_table[b, c]
            ca = qr_table[c, a]
            
            # If ab, bc, ca are all True, it's a cycle a->b->c->a
            # If ab, cb, ac are all True, it's a cycle a->c->b->a
            # A tournament on 3 vertices is cyclic if it contains a 3-cycle.
            # Edges: (a,b), (b,c), (a,c).
            # Cycle 1: a->b, b->c, c->a. (ab, bc, ca)
            # Cycle 2: a->c, c->b, b->a. (ac, cb, ba) -> (!ab, !bc, !ca)
            
            is_cyclic = (ab and bc and ca) or (not ab and not bc and not ca)
            
            if is_cyclic:
                colors[i] = 0 # Red
            else:
                colors[i] = 1 # Blue
                
        return colors, triples, idx, m

    def _repair_vectorized(colors, quads, pents, time_budget):
        """
        Vectorized repair.
        Red K4: 4 triples all 0.
        Blue K5: 10 triples all 1.
        
        We will iterate through triples and try to flip them if it reduces violations.
        To make it fast, we precompute which quads/pents each triple belongs to.
        """
        m = len(colors)
        colors = colors.copy()
        
        # Precompute incidence
        tri_quads = [[] for _ in range(m)]
        for qi, idxs in enumerate(quads):
            for i in idxs:
                tri_quads[i].append(qi)
                
        tri_pents = [[] for _ in range(m)]
        for pi, idxs in enumerate(pents):
            for i in idxs:
                tri_pents[i].append(pi)
        
        # Count current violations
        quad_red_counts = np.zeros(len(quads), dtype=np.int8)
        for qi, idxs in enumerate(quads):
            quad_red_counts[qi] = np.sum(colors[idxs] == 0)
            
        pent_blue_counts = np.zeros(len(pents), dtype=np.int8)
        for pi, idxs in enumerate(pents):
            pent_blue_counts[pi] = np.sum(colors[idxs] == 1)
            
        red_k4 = np.sum(quad_red_counts == 4)
        blue_k5 = np.sum(pent_blue_counts == 10)
        
        if red_k4 == 0 and blue_k5 == 0:
            return colors, red_k4, blue_k5
            
        t_end = time.time() + time_budget
        order = np.arange(m)
        np.random.shuffle(order)
        
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for t in order:
                if time.time() >= t_end:
                    break
                cur = colors[t]
                new = 1 - cur
                delta_red = 1 if new == 0 else -1
                delta_blue = 1 if new == 1 else -1
                
                dq = 0
                for qi in tri_quads[t]:
                    old = quad_red_counts[qi]
                    new_q = old + delta_red
                    if old == 4: dq -= 1
                    if new_q == 4: dq += 1
                    
                dp = 0
                for pi in tri_pents[t]:
                    old = pent_blue_counts[pi]
                    new_p = old + delta_blue
                    if old == 10: dp -= 1
                    if new_p == 10: dp += 1
                    
                if dq + dp < 0:
                    colors[t] = new
                    for qi in tri_quads[t]:
                        quad_red_counts[qi] += delta_red
                    for pi in tri_pents[t]:
                        pent_blue_counts[pi] += delta_blue
                    red_k4 += dq
                    blue_k5 += dp
                    improved = True
            np.random.shuffle(order)
            
        return colors, red_k4, blue_k5

    # Try n values from high to low
    # Known best is 34. Let's try 34, 33, 32...
    # Paley construction works best for n = 4k+1 or similar.
    # 33 is not prime. 32 is not prime. 31 is prime.
    # Let's try n=31, 32, 33, 34 with different initializations if Paley fails.
    
    deadline = _START + _DEADLINE - 5.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try n=34 first with a random start + repair, then Paley for primes
    # Actually, the task says "published construction has 34 points".
    # Let's try to find it via search.
    
    for n in [34, 33, 32, 31, 30, 29, 28]:
        if time.time() >= deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        # Strategy A: Paley-like for prime n
        if n == 31:
            colors, triples, idx, m = _paley_initial(n)
            quads = _build_quads(n)
            pents = _build_pents(n)
            colors, r4, b5 = _repair_vectorized(colors, quads, pents, remaining - 1.0)
            if r4 == 0 and b5 == 0:
                best_n = n
                best_colors = colors.tolist()
                continue # Found valid for this n, try larger? No, we iterate high to low.
                # But if we found 31, we should have tried 34 first.
                # So if we are at 31, 34-32 failed.
        
        # Strategy B: Random start + repair for all n
        # Use multiple restarts if time permits
        attempts = 0
        max_attempts = max(1, int(remaining / 2.0))
        success = False
        
        for _ in range(max_attempts):
            if time.time() >= deadline - 1.0:
                break
            triples, idx, m = _triples_to_idx(n)
            quads = _build_quads(n)
            pents = _build_pents(n)
            
            # Random start with bias
            # Red K4 is smaller, so maybe bias towards Blue?
            # Or balanced.
            colors = (np.random.rand(m) < 0.5).astype(np.int8)
            
            colors, r4, b5 = _repair_vectorized(colors, quads, pents, remaining / max_attempts - 0.5)
            
            if r4 == 0 and b5 == 0:
                best_n = n
                best_colors = colors.tolist()
                success = True
                break
                
        if success:
            # If we found a valid coloring for n, we can stop if n is high enough
            # Or continue to try larger? We are iterating high to low.
            # So if we found 34, we are done.
            # If we found 33, we can stop.
            break
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
