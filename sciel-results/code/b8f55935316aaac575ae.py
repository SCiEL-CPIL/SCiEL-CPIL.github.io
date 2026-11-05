"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_valid_set(S, limit=512):
    """Check if all pairwise lcms in S are <= limit."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if _lcm(S[i], S[j]) > limit:
                return False
    return True

def _divisors_of(n):
    """Return all divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin under 90s limit
    N = 512
    
    # Precompute lcm table for speed
    # lcm_table[a-1][b-1] = lcm(a,b)
    lcm_table = [[0] * N for _ in range(N)]
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            lcm_table[a-1][b-1] = _lcm(a, b)
    
    best_size = 0
    best_set = set([1])
    
    def try_greedy_augment(S_set, S_mask):
        """Try to add elements greedily to S_set."""
        S_set = set(S_set)
        S_mask = 0
        for x in S_set:
            S_mask |= (1 << (x - 1))
        
        # Repeatedly try to add any compatible element
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S_set:
                    # Check if x is compatible with all elements in S_set
                    compatible = True
                    for s in S_set:
                        if lcm_table[x-1][s-1] > N:
                            compatible = False
                            break
                    if compatible:
                        S_set.add(x)
                        S_mask |= (1 << (x - 1))
                        changed = True
        
        return S_set, S_mask
    
    def local_search(S_set, max_swaps=100):
        """Try to improve the set via swaps."""
        S_set = set(S_set)
        current_size = len(S_set)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            improved = False
            # Try removing one element and adding another
            for u in list(S_set):
                S_wo_u = S_set - {u}
                # Find best v not in S_set that is compatible with S_wo_u
                best_v = None
                best_gain = 0
                for v in range(1, N + 1):
                    if v in S_set:
                        continue
                    # Check if v is compatible with all in S_wo_u
                    compatible = True
                    for s in S_wo_u:
                        if lcm_table[v-1][s-1] > N:
                            compatible = False
                            break
                    if compatible:
                        # This swap maintains validity; we just swap one for another
                        # Only useful if we can then add more
                        if best_v is None:
                            best_v = v
                        break  # Just take first valid swap for now
                
                if best_v is not None:
                    S_set = S_wo_u | {best_v}
                    # Try to augment after swap
                    S_set, _ = try_greedy_augment(S_set, 0)
                    if len(S_set) > current_size:
                        current_size = len(S_set)
                        improved = True
                        break
        
        return S_set
    
    # Strategy 1: For each M, start with divisors of M, then augment
    # Sort M by number of divisors (descending) to find good candidates first
    candidates_M = []
    for M in range(1, N + 1):
        divs = _divisors_of(M)
        candidates_M.append((len(divs), M, divs))
    
    # Sort by number of divisors descending
    candidates_M.sort(key=lambda x: (-x[0], x[1]))
    
    # Also include some random M values
    random.seed(42)
    extra_Ms = random.sample(range(1, N + 1), min(50, N))
    
    processed_Ms = set()
    
    for num_divs, M, divs in candidates_M:
        if time.time() > deadline:
            break
        if M in processed_Ms:
            continue
        processed_Ms.add(M)
        
        # Start with divisors of M
        S_set = set(divs)
        S_set, _ = try_greedy_augment(S_set, 0)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
        
        # Try local search
        S_improved = local_search(S_set, max_swaps=20)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = set(S_improved)
    
    # Strategy 2: Try random initial sets and augment
    for _ in range(30):
        if time.time() > deadline:
            break
        # Random set of size 10-20
        size = random.randint(10, 20)
        S_set = set(random.sample(range(1, N + 1), size))
        S_set, _ = try_greedy_augment(S_set, 0)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
        
        # Local search
        S_improved = local_search(S_set, max_swaps=15)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = set(S_improved)
    
    # Final polish: try to add any remaining compatible elements
    final_set, _ = try_greedy_augment(best_set, 0)
    if len(final_set) > best_size:
        best_size = len(final_set)
        best_set = final_set
    
    result = sorted(best_set)
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
