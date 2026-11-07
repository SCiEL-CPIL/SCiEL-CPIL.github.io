"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Compute LCM of all divisors
LCM = 1
for d in DIVISORS:
    LCM = _lcm(LCM, d)

def construct():
    """
    Find 10 distinct positive integers <= 1000000, each divisible by its corresponding
    divisor in DIVISORS, minimizing max - min.
    
    Uses the observation that if we have a window [S, S+R], we need to pick one multiple
    of DIVISORS[i] in [S, S+R] for each i, all distinct.
    
    Key insight: The pattern of available multiples repeats modulo LCM. So we can
    search over window start positions modulo LCM and window sizes.
    """
    # Precompute: for each divisor d, the set of residues mod LCM that are multiples of d
    # A number x is a multiple of d iff x % d == 0.
    # In a window [S, S+R], the multiples of d are those x in [S, S+R] with x % d == 0.
    # The number of such multiples is either floor(R/d) or floor(R/d)+1.
    
    # For feasibility: we need to assign to each divisor i a distinct multiple of DIVISORS[i]
    # within [S, S+R]. This is a bipartite matching problem, but with the structure that
    # each "slot" (a specific number in the window) can serve multiple divisors.
    
    # Better approach: For a given window size R and start offset s (mod LCM), 
    # check if there exists an assignment of distinct numbers.
    
    # Let's think differently. For a window [S, S+R], for each divisor d_i, the set of
    # valid numbers is {S + k : 0 <= k <= R, (S+k) % d_i == 0}.
    # We need to pick one from each set, all distinct.
    
    # This is a constraint satisfaction problem. Given the small size (10 divisors),
    # we can use backtracking with pruning.
    
    # But first, let's find a good R. The minimum possible R is at least 1 (since distinct).
    # Let's search for the minimum R such that there exists S with a valid assignment.
    
    # Since the pattern repeats mod LCM, we only need to check S mod LCM.
    # But R can be up to... well, the window can be large. Let's start small.
    
    # For a given R and S mod LCM, let's check feasibility.
    # For each divisor i, compute the list of offsets k in [0, R] such that (S+k) % d_i == 0.
    # Then we need to pick distinct offsets... wait, no. The actual numbers are S+k, and they
    # must be distinct. Since S is fixed, distinct k means distinct numbers.
    
    # So we need: for each i, pick k_i from K_i = {k in [0,R] : (S+k) % d_i == 0},
    # such that all k_i are distinct.
    
    # This is a matching problem: left side is divisors (10 nodes), right side is offsets {0,...,R}.
    # Each divisor i can be matched to any k in K_i.
    
    # For small R, we can do this with backtracking or Hungarian algorithm.
    # Let's try increasing R from 0 upward, and for each R, try all S mod LCM.
    
    # Optimization: R must be at least 1. Let's start from R=1.
    
    # Actually, let's be smarter. For a given R, the number of multiples of d in a window of
    # size R+1 (from S to S+R inclusive) is either floor((R+1)/d) or ceil((R+1)/d).
    # Specifically, it's either floor(R/d) or floor(R/d)+1.
    
    # For the assignment to be possible, we need at least 1 multiple of each d_i in the window,
    # and the Hall's condition must hold for the bipartite graph.
    
    # Let's just do a direct search. Start with R=1, increase until we find a feasible (R, S).
    
    # Precompute for each divisor d, the residues mod LCM where x % d == 0.
    # Actually, (S + k) % d == 0 iff k % d == (-S) % d.
    # So for a given S and divisor d, the valid k are those with k ≡ (-S) mod d.
    
    # Let's implement the feasibility check.
    
    def is_feasible(R, S_mod):
        """Check if window [S, S+R] with S ≡ S_mod (mod LCM) has a valid assignment."""
        # For each divisor i, compute the set of valid k values in [0, R]
        # k must satisfy (S_mod + k) % DIVISORS[i] == 0
        # i.e., k ≡ (-S_mod) % DIVISORS[i] (mod DIVISORS[i])
        
        valid_sets = []
        for i, d in enumerate(DIVISORS):
            remainder = (-S_mod) % d
            # k in [0, R] with k % d == remainder
            ks = []
            k = remainder
            if k > R:
                # No valid k
                return False
            while k <= R:
                ks.append(k)
                k += d
            if not ks:
                return False
            valid_sets.append(ks)
        
        # Now we need to assign distinct k values to each divisor.
        # Use backtracking with ordering by smallest set first.
        order = sorted(range(N), key=lambda i: len(valid_sets[i]))
        
        used = set()
        
        def backtrack(idx):
            if idx == N:
                return True
            i = order[idx]
            for k in valid_sets[i]:
                if k not in used:
                    used.add(k)
                    if backtrack(idx + 1):
                        return True
                    used.remove(k)
            return False
        
        return backtrack(0)
    
    # Search for minimum R
    # R starts at 1 (since we need distinct values, window must have at least 2 numbers... 
    # wait, no. R is max - min, so if min=S and max=S+R, the window has R+1 integers.
    # We need 10 distinct integers, so R+1 >= 10, i.e., R >= 9.
    
    # Actually wait: we need 10 distinct positive integers. They all lie in [S, S+R].
    # The number of integers in [S, S+R] is R+1. We need R+1 >= 10, so R >= 9.
    
    # But actually, we don't need ALL integers in the window to be used, just 10 of them.
    # So R >= 9 is a necessary condition.
    
    for R in range(9, 1000):  # Upper bound is generous
        for S_mod in range(LCM):
            if is_feasible(R, S_mod):
                # Found a feasible window. Now find the actual assignment.
                # S can be S_mod + t*LCM for some t >= 0, such that S >= 1 and S+R <= 1000000.
                # We need S >= 1, so t*LCM + S_mod >= 1.
                # And S + R <= 1000000, so t*LCM + S_mod + R <= 1000000.
                
                # Find smallest valid S
                S = S_mod
                if S < 1:
                    # Need S >= 1
                    # S = S_mod + t*LCM >= 1
                    # t >= ceil((1 - S_mod)/LCM)
                    if S_mod >= 1:
                        S = S_mod
                    else:
                        S = LCM  # t=1 gives S=LCM
                        # Actually let's compute properly
                        t_min = (1 - S_mod + LCM - 1) // LCM if S_mod < 1 else 0
                        S = S_mod + t_min * LCM
                
                # Check S + R <= 1000000
                if S + R > 1000000:
                    continue
                
                # Now find the actual assignment
                valid_sets = []
                for i, d in enumerate(DIVISORS):
                    remainder = (-S_mod) % d
                    ks = []
                    k = remainder
                    while k <= R:
                        ks.append(k)
                        k += d
                    valid_sets.append(ks)
                
                order = sorted(range(N), key=lambda i: len(valid_sets[i]))
                used = set()
                assignment = {}
                
                def backtrack2(idx):
                    if idx == N:
                        return True
                    i = order[idx]
                    for k in valid_sets[i]:
                        if k not in used:
                            used.add(k)
                            assignment[i] = k
                            if backtrack2(idx + 1):
                                return True
                            used.remove(k)
                            del assignment[i]
                    return False
                
                if backtrack2(0):
                    result = [S + assignment[i] for i in range(N)]
                    # Verify all constraints
                    assert all(isinstance(x, int) for x in result)
                    assert len(set(result)) == N
                    assert all(1 <= x <= 1000000 for x in result)
                    assert all(result[i] % DIVISORS[i] == 0 for i in range(N))
                    return list(result)
    
    # Fallback: just return the divisors themselves (which is valid but not optimal)
    return DIVISORS[:]

# EVOLVE-BLOCK-END
