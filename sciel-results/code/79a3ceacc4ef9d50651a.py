"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    n = 128
    N = 128  # elements 1..128, indexed 0..127
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute all bad triples: for each pair (a,b), find all c such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # conflict[a][b] = set of c values that form a bad triple with a and b
    # We'll use: for element x, conflict_mask[x] is a bitmask of elements y
    # such that there exists z where {x,y,z} is a bad triple.
    # Actually, we need: when adding x to a set S, check if there exist a,b in S
    # such that {a,b,x} is a bad triple.
    
    # Better: precompute bad_pairs[x] = set of (a,b) pairs such that {a,b,x} is bad
    # Even better: for each x, precompute conflict_with[x] = set of y such that
    # there exists z (z != x, z != y) where {x,y,z} is a bad triple.
    # But this isn't quite right either - we need the specific triple.
    
    # Most efficient: precompute for each element x, a bitmask conflict_mask[x]
    # where bit y is set if there exists some z such that {x,y,z} is a bad triple.
    # Wait, that's not sufficient. We need: when x is in the set, for any pair (a,b)
    # already in the set, check if {a,b,x} is bad.
    
    # Let's precompute: bad_triple_exists[a][b][x] = True if {a,b,x} is a bad triple
    # But that's O(n^3) = 2M entries, fine.
    
    # Actually, for fast checking: when we have a set S (as bitmask), and want to
    # add x, we need to check: for all pairs (a,b) in S, is {a,b,x} a bad triple?
    # Equivalently: is there any a in S such that (a,x) is in the "bad pair" relation
    # with some b also in S?
    
    # Precompute: for each pair (x, a), the set of b such that {x,a,b} is a bad triple.
    # Then when adding x to S: for each a in S, check if bad_partners[x][a] & S != 0.
    
    # bad_partners[x][a] = bitmask of b values where {x,a,b} is a bad triple
    # x, a in 1..128, so we use 0-indexed: x-1, a-1
    bad_partners = [[0] * N for _ in range(N)]  # bad_partners[x-1][a-1] = bitmask of b-1
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    # {a,b,c} is a bad triple
                    bad_partners[a-1][b-1] |= (1 << (c-1))
                    bad_partners[a-1][c-1] |= (1 << (b-1))
                    bad_partners[b-1][a-1] |= (1 << (c-1))
                    bad_partners[b-1][c-1] |= (1 << (a-1))
                    bad_partners[c-1][a-1] |= (1 << (b-1))
                    bad_partners[c-1][b-1] |= (1 << (a-1))
    
    def can_add(x, mask):
        """Check if element x (1-indexed) can be added to set represented by bitmask mask."""
        xi = x - 1
        # For each a in the current set, check if bad_partners[xi][a-1] & mask != 0
        # But we need to be careful: we're checking if adding x creates a bad triple.
        # A bad triple {x, a, b} exists if a is in set AND b is in set AND {x,a,b} is bad.
        # bad_partners[xi][a-1] gives all b such that {x,a,b} is bad.
        # So we need: for some a in set, (bad_partners[xi][a-1] & mask) != 0
        # But also a != x (x not in set yet, so a != x automatically)
        # And b != x, b != a (b in mask means b is in set, and b != x since x not in set)
        m = mask
        while m:
            lb = m & (-m)
            ai = lb.bit_length() - 1
            if (bad_partners[xi][ai] & mask) != 0:
                return False
            m ^= lb
        return True
    
    def is_valid_mask(mask):
        """Check if the set represented by mask is valid (no bad triples)."""
        # For each pair (a,b) in set, check if bad_partners[a-1][b-1] & mask != 0
        # (excluding a and b themselves from the check, but they won't be in bad_partners
        #  for their own pair since we only stored distinct elements)
        m = mask
        while m:
            la = m & (-m)
            ai = la.bit_length() - 1
            m ^= la
            m2 = m
            while m2:
                lb = m2 & (-m2)
                bi = lb.bit_length() - 1
                m2 ^= lb
                # Check if there's a c in mask (c != a, c != b) such that {a,b,c} is bad
                # bad_partners[ai][bi] has bits for all c where {a+1,b+1,c+1} is bad
                # We need c in mask, c != a, c != b
                bp = bad_partners[ai][bi]
                # Remove bits for a and b if present
                bp &= ~(la | lb)
                if (bp & mask) != 0:
                    return False
        return True
    
    # Greedy construction with multiple orderings
    def greedy(order):
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Try multiple greedy orderings
    orderings = [
        list(range(1, n + 1)),           # natural order
        list(range(n, 0, -1)),           # reverse order
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),  # high bit count first
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),   # low bit count first
    ]
    
    for order in orderings:
        mask = greedy(order)
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Simulated Annealing
    def sa_from(mask, max_iters=50000):
        """Simulated annealing starting from mask."""
        current_mask = mask
        current_size = current_mask.bit_count()
        best_local = current_mask
        best_local_size = current_size
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9997
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Add an element
                not_in = ((1 << N) - 1) & ~current_mask
                if not_in == 0:
                    continue
                # Pick random element not in set
                m = not_in
                # Pick random set bit
                idx = random.randint(0, m.bit_count() - 1)
                # Find the idx-th set bit
                tmp = m
                for _ in range(idx):
                    tmp &= tmp - 1
                x = (tmp & (-tmp)).bit_length()  # 1-indexed
                if can_add(x, current_mask):
                    new_mask = current_mask | (1 << (x - 1))
                    new_size = new_size = new_mask.bit_count()
                    delta = new_size - current_size
                    if delta > 0 or random.random() < (1.0 if T > T_min else 0):
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local = current_mask
                else:
                    # Can't add, skip
                    pass
            elif move_type < 0.5:
                # Remove an element
                if current_mask == 0:
                    continue
                m = current_mask
                idx = random.randint(0, m.bit_count() - 1)
                tmp = m
                for _ in range(idx):
                    tmp &= tmp - 1
                x = (tmp & (-tmp)).bit_length()  # 1-indexed
                new_mask = current_mask & ~(1 << (x - 1))
                new_size = new_mask.bit_count()
                delta = new_size - current_size
                if delta > 0 or random.random() < (T if T > T_min else 0.001):
                    current_mask = new_mask
                    current_size = new_size
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local = current_mask
            else:
                # Swap: remove one, add another
                if current_mask == 0:
                    continue
                m_in = current_mask
                idx_in = random.randint(0, m_in.bit_count() - 1)
                tmp_in = m_in
                for _ in range(idx_in):
                    tmp_in &= tmp_in - 1
                x_remove = (tmp_in & (-tmp_in)).bit_length()  # 1-indexed
                
                not_in = ((1 << N) - 1) & ~current_mask
                if not_in == 0:
                    continue
                m_out = not_in
                idx_out = random.randint(0, m_out.bit_count() - 1)
                tmp_out = m_out
                for _ in range(idx_out):
                    tmp_out &= tmp_out - 1
                x_add = (tmp_out & (-tmp_out)).bit_length()  # 1-indexed
                
                new_mask = current_mask & ~(1 << (x_remove - 1))
                if can_add(x_add, new_mask):
                    new_mask |= (1 << (x_add - 1))
                    new_size = new_mask.bit_count()
                    delta = new_size - current_size
                    if delta > 0 or random.random() < (T if T > T_min else 0.001):
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local = current_mask
            
            T *= cooling
        
        return best_local, best_local_size
    
    # Run SA from best greedy solution and a few random starts
    if time.time() < deadline:
        mask, sz = sa_from(best_mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Try SA from a few random initial solutions
    random_starts = 3
    for _ in range(random_starts):
        if time.time() >= deadline - 5:
            break
        # Random initial set
        init_mask = 0
        for x in range(1, n + 1):
            if random.random() < 0.5:
                if can_add(x, init_mask):
                    init_mask |= (1 << (x - 1))
        mask, sz = sa_from(init_mask, max_iters=30000)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Final local search: try to add any element
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask & (1 << (x - 1))):
                    if can_add(x, best_mask):
                        best_mask |= (1 << (x - 1))
                        best_size += 1
                        improved = True
    
    # Convert mask to list
    result = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
    
    # Final verification
    assert len(result) > 0
    for a, b, c in combinations(result, 3):
        assert not (_lcm(a, b) == _lcm(a, c) == _lcm(b, c)), f"Bad triple: {a},{b},{c}"
    
    return result
# EVOLVE-BLOCK-END
