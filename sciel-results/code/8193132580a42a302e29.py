"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank(n, m, p, target_rank, deadline, seed):
    """
    Attempt to find a decomposition of rank 'target_rank' using random search.
    We use a simple local search / random restart approach with rational numbers
    (specifically integers and half-integers as allowed, or just integers for simplicity 
    and then scaling if needed, but the prompt says integer/half-integer).
    
    Actually, for small ranks, we can use a more structured approach if we know specific patterns,
    but random search over a small grid of rational values (like -1, 0, 1) is a good heuristic.
    """
    rng = random.Random(seed)
    
    # We represent U, V, W as lists of lists of Fractions to ensure exactness
    # But for speed, we'll try integer values first: -1, 0, 1
    
    # The condition is:
    # For all i, k, j1, j2: sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] == delta(j1, j2)
    
    # Let's try a random initialization with values in {-1, 0, 1}
    # And then do a local search (flip bits) to satisfy constraints.
    
    # Number of constraints: n * p * m * m = 2 * 3 * 3 * 3 = 54
    # Number of variables: 3 * target_rank * (n*m + m*p + p*n) / 3? No.
    # Variables: (n*m + m*p + p*n) * target_rank
    # For n=2, m=3, p=3: (6 + 9 + 6) * R = 21 * R variables.
    # For R=18, 378 variables.
    # For R=17, 357 variables.
    
    # This is an underdetermined system if R is large enough?
    # 21 * R >= 54 => R >= 2.57. So R=3 is theoretically possible if linearly independent enough.
    
    # However, the structure is specific.
    
    # Strategy: Randomly initialize U, V, W with small integers.
    # Check which constraints are violated.
    # Pick a violated constraint and a variable involved, perturb it.
    
    # Since the space is discrete, we can try a simple random restart with local search.
    
    max_iters = 5000
    values = [-1, 0, 1]
    
    for attempt in range(max_iters):
        if time.time() > deadline:
            return None
            
        # Initialize
        U = [[rng.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Check initial
        if _verify(U, V, W, n, m, p):
            return U, V, W
            
        # Local search: 
        # We can't easily do gradient descent on discrete variables.
        # Let's just try more random restarts.
        
        # Optimization: We can check constraints incrementally or just do full check.
        # For R=17, full check is 54 * 17 = 918 multiplications. Very fast.
        # 5000 restarts * 918 ops = ~4.5M ops. Should take < 1s in Python.
        
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook (rank 18)
    U_best, V_best, W_best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Lower bound is max(n*m, m*p, n*p) = 9.
    # But known rank for 2x3x3 is likely 18. Let's check 17, 16, etc.
    
    # We will search from best_rank - 1 downwards.
    # If we find one, we update best.
    
    # To maximize chance, we try a few ranks.
    # Given the time budget, we can try many restarts for rank 17.
    
    # Let's try rank 17 first.
    # Then 16, etc.
    
    # Note: The problem asks for rank-minimal.
    # If we don't find < 18, we return 18.
    
    # We'll allocate time to search for lower ranks.
    # If we find a valid decomposition for rank R, we return it.
    
    # Let's try ranks 17 down to 9.
    # We give each rank a portion of the time.
    
    # However, random search over {-1,0,1} might miss solutions that require 2 or 1/2.
    # But typically minimal rank decompositions can be found with small integers.
    
    # Let's expand search space to {-2, -1, 0, 1, 2} for a few attempts if simple fails?
    # No, keep it simple first.
    
    seed = 42
    for target_rank in range(best_rank - 1, 8, -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find solution for this rank
        # We'll do a limited number of restarts
        # More restarts for lower ranks if we have time?
        # Actually, lower ranks are harder, so we might want to spend less time if we don't find quickly?
        # Or more time?
        
        # Let's just do a fixed number of attempts per rank, say 2000.
        res = _try_rank(n, m, p, target_rank, deadline, seed)
        if res:
            U_best, V_best, W_best = res
            best_rank = target_rank
            # If we found a lower rank, we might want to keep searching for even lower?
            # Yes, the task is rank-minimal.
            # But if we found 17, we should try 16.
            continue
        
        seed += 1
        
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
