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
    
    # Precompute bad triples conflicts
    # bad_partners[x-1][a-1] is a bitmask of b-1 such that {x, a, b} is a bad triple.
    # Note: We only store b > a, b > x, etc? No, we store all b != x, b != a.
    # Actually, for a specific pair (x, a), the set of b such that {x, a, b} is bad is fixed.
    bad_partners = [[0] * N for _ in range(N)]
    
    # Identify all bad triples
    # Triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # We can iterate a<b<c
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            # c must be such that lcm(a,c) == L_ab and lcm(b,c) == L_ab
            # This implies c divides L_ab.
            # Also c > b.
            # We can just iterate c from b+1 to n
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    # Found bad triple {a,b,c}
                    # Update bad_partners for all 3 pairs
                    # Pair (a,b) conflicts with c
                    bad_partners[a-1][b-1] |= (1 << (c-1))
                    bad_partners[b-1][a-1] |= (1 << (c-1))
                    # Pair (a,c) conflicts with b
                    bad_partners[a-1][c-1] |= (1 << (b-1))
                    bad_partners[c-1][a-1] |= (1 << (b-1))
                    # Pair (b,c) conflicts with a
                    bad_partners[b-1][c-1] |= (1 << (a-1))
                    bad_partners[c-1][b-1] |= (1 << (a-1))

    def can_add(x, mask):
        """Check if element x (1-indexed) can be added to set represented by bitmask mask."""
        xi = x - 1
        # Check against all a in mask
        m = mask
        while m:
            lb = m & (-m)
            ai = lb.bit_length() - 1
            # If there exists b in mask such that {x, a, b} is bad, return False
            # bad_partners[xi][ai] contains all such b.
            # We need to check if any such b is in mask.
            # Note: b must be distinct from x and a.
            # Since x is not in mask yet, and a is in mask, b is in mask.
            # If b == x, it's not in mask. If b == a, it's not in bad_partners[xi][ai] 
            # because bad triples have distinct elements.
            if (bad_partners[xi][ai] & mask) != 0:
                return False
            m ^= lb
        return True

    def is_valid_mask(mask):
        """Verify the mask is valid. Used for final check and debugging."""
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
                # Check if any c in mask (c != a, c != b) completes a bad triple
                bp = bad_partners[ai][bi]
                # Exclude a and b from bp just in case (though they shouldn't be there)
                bp &= ~(la | lb)
                if (bp & mask) != 0:
                    return False
        return True

    # Helper to get a random set bit index from a mask
    def get_random_bit(mask):
        if mask == 0:
            return -1
        idx = random.randint(0, mask.bit_count() - 1)
        tmp = mask
        for _ in range(idx):
            tmp &= tmp - 1
        return (tmp & (-tmp)).bit_length() - 1 # 0-indexed

    # Greedy construction
    def greedy(order):
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        return mask

    best_mask = 0
    best_size = 0
    
    # Define various orderings
    nums = list(range(1, n + 1))
    
    # 1. Natural
    orders = [
        nums,
        nums[::-1],
        sorted(nums, key=lambda x: x.bit_count()),
        sorted(nums, key=lambda x: -x.bit_count()),
        sorted(nums, key=lambda x: x),
        sorted(nums, key=lambda x: -x),
    ]
    
    # 2. By number of prime factors (approximated by bit_count for now, or compute omega)
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
        
    omega_cache = {x: omega(x) for x in nums}
    orders.append(sorted(nums, key=lambda x: omega_cache[x]))
    orders.append(sorted(nums, key=lambda x: -omega_cache[x]))
    
    # 3. Randomized shuffles
    for _ in range(5):
        rand_order = nums[:]
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
        mask = greedy(order)
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Simulated Annealing / Local Search
    def local_search(start_mask, max_iters=100000):
        current_mask = start_mask
        current_size = current_mask.bit_count()
        best_local_mask = current_mask
        best_local_size = current_size
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9998
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                not_in = ((1 << N) - 1) & ~current_mask
                if not_in:
                    x_idx = get_random_bit(not_in)
                    if x_idx != -1:
                        x = x_idx + 1
                        if can_add(x, current_mask):
                            new_mask = current_mask | (1 << x_idx)
                            new_size = new_mask.bit_count()
                            delta = new_size - current_size
                            if delta > 0 or random.random() < (T if T > T_min else 0.0):
                                current_mask = new_mask
                                current_size = new_size
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_mask = current_mask
            elif move_type < 0.6:
                # Remove
                if current_mask:
                    x_idx = get_random_bit(current_mask)
                    if x_idx != -1:
                        new_mask = current_mask & ~(1 << x_idx)
                        new_size = new_mask.bit_count()
                        delta = new_size - current_size
                        if delta > 0 or random.random() < (T if T > T_min else 0.001):
                            current_mask = new_mask
                            current_size = new_size
                            # Update best only if size increased? No, we track best size.
                            # Since we removed, size decreased, so best_local won't update unless we later add.
            else:
                # Swap (1-for-1)
                if current_mask:
                    x_idx = get_random_bit(current_mask)
                    not_in = ((1 << N) - 1) & ~current_mask
                    if x_idx != -1 and not_in:
                        y_idx = get_random_bit(not_in)
                        if y_idx != -1:
                            # Try swapping x for y
                            temp_mask = current_mask & ~(1 << x_idx)
                            y = y_idx + 1
                            if can_add(y, temp_mask):
                                new_mask = temp_mask | (1 << y_idx)
                                new_size = new_mask.bit_count()
                                delta = new_size - current_size
                                if delta > 0 or random.random() < (T if T > T_min else 0.01):
                                    current_mask = new_mask
                                    current_size = new_size
                                    if current_size > best_local_size:
                                        best_local_size = current_size
                                        best_local_mask = current_mask
            T *= cooling
            
        return best_local_mask, best_local_size

    # Run LS from best greedy
    if time.time() < deadline:
        mask, sz = local_search(best_mask, max_iters=50000)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Run LS from a few random valid starts
    for _ in range(3):
        if time.time() >= deadline - 10:
            break
        # Generate random valid start
        init_mask = 0
        rand_order = nums[:]
        random.shuffle(rand_order)
        for x in rand_order:
            if can_add(x, init_mask):
                init_mask |= (1 << (x - 1))
        
        mask, sz = local_search(init_mask, max_iters=30000)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Final polish: try to add any single element
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
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
