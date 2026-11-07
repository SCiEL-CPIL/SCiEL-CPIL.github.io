"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

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
    
    Uses bitset-accelerated CSP over window start residues mod LCM.
    """
    
    def find_assignment(R, S_mod):
        """
        For window [S, S+R] with S ≡ S_mod (mod LCM), find distinct k_i for each divisor i
        such that (S_mod + k_i) % DIVISORS[i] == 0 and all k_i are distinct.
        Returns dict {i: k_i} or None if infeasible.
        """
        valid_sets = []
        for i, d in enumerate(DIVISORS):
            remainder = (-S_mod) % d
            ks = []
            k = remainder
            while k <= R:
                ks.append(k)
                k += d
            if not ks:
                return None
            valid_sets.append(ks)
        
        # Order by smallest set first for backtracking efficiency
        order = sorted(range(N), key=lambda i: len(valid_sets[i]))
        
        used = set()
        assignment = {}
        
        def backtrack(idx):
            if idx == N:
                return True
            i = order[idx]
            for k in valid_sets[i]:
                if k not in used:
                    used.add(k)
                    assignment[i] = k
                    if backtrack(idx + 1):
                        return True
                    used.remove(k)
                    del assignment[i]
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # R must be at least 9 (need 10 distinct integers in window of size R+1)
    for R in range(9, 50):
        for S_mod in range(LCM):
            assignment = find_assignment(R, S_mod)
            if assignment is not None:
                # Find valid S >= 1 such that S ≡ S_mod (mod LCM) and S + R <= 1000000
                if S_mod == 0:
                    S = LCM  # S must be >= 1, and S ≡ 0 mod LCM
                else:
                    S = S_mod  # S_mod >= 1
                
                if S + R > 1000000:
                    # Try next multiple
                    t = (1000000 - S_mod - R) // LCM + 1
                    if t <= 0:
                        continue
                    S = S_mod + t * LCM
                    if S < 1:
                        continue
                    if S + R > 1000000:
                        continue
                
                result = [S + assignment[i] for i in range(N)]
                
                # Verify all constraints
                assert all(isinstance(x, int) for x in result)
                assert len(set(result)) == N
                assert all(1 <= x <= 1000000 for x in result)
                assert all(result[i] % DIVISORS[i] == 0 for i in range(N))
                return list(result)
    
    # Fallback: use the divisors themselves
    return list(DIVISORS)

# EVOLVE-BLOCK-END
