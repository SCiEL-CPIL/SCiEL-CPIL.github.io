import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140

    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    target_set = set(target_indices)

    def schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        for idx, (a, b, c) in enumerate(target_indices):
            U[a, idx] = 1
            V[b, idx] = 1
            W[c, idx] = 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank_schoolbook

    def verify(U, V, W, rank, tol=1e-9):
        for (a, b, c) in target_indices:
            s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
            if abs(s - 1.0) > tol:
                return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target_set:
                        s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    # Structural approach: start from schoolbook and try to merge terms
    # by finding linear dependencies. Use a smarter local search with
    # structured initialization based on the tensor's symmetry.
    
    def try_merge(U, V, W, rank):
        """Try to merge two terms in the decomposition to reduce rank by 1."""
        for r1 in range(rank):
            if time.time() > deadline - 10:
                return None
            for r2 in range(r1 + 1, rank):
                # Check if terms r1 and r2 can be merged
                # They can be merged if their support patterns allow it
                u1, u2 = U[:, r1], U[:, r2]
                v1, v2 = V[:, r1], V[:, r2]
                w1, w2 = W[:, r1], W[:, r2]
                
                # For merging, we need to find a single term that
                # reproduces the contribution of both terms
                # This is possible if the two terms have overlapping structure
                
                # Try: new_u = u1 + u2, new_v = v1, new_w = w1 + w2
                # This works if u1*v1*w1 + u2*v2*w2 = (u1+u2)*v1*(w1+w2)
                # which requires u1*v1*w2 + u2*v1*w1 + u2*v2*w2 = 0 for all (a,b,c)
                
                # Simpler: try to find alpha, beta such that
                # u1*v1*w1 + u2*v2*w2 = (u1 + alpha*u2)*v1*(w1 + beta*w2)
                # This is complex. Instead, use a direct approach:
                
                # Try setting one of the vectors to zero pattern
                # and adjusting the others
                pass
        return None

    # Better structural approach: use the fact that for n=2, the two
    # output rows share structure. Try a parametrized family of solutions.
    
    # For 2x3x3, try rank 15 first with a structured initialization
    # where we exploit the j-index sharing between U and V
    
    def structured_init(target_rank):
        """Create a structured initialization that respects the tensor's symmetry."""
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Assign terms to (i, j, k) triples, possibly sharing
        # For rank < 18, we need to share some terms
        # Strategy: for each j, try to share the V component across i and k
        
        # Simple structured init: assign terms greedily
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        
        # Shuffle and assign to rank slots
        random.shuffle(terms)
        for idx, (i, j, k) in enumerate(terms):
            if idx >= target_rank:
                break
            U[i * m + j, idx] = 1.0
            V[j * p + k, idx] = 1.0
            W[k * n + i, idx] = 1.0
        
        return U, V, W

    # Use coordinate descent with multiple restarts and structured inits
    for target_rank in range(rank_schoolbook - 1, 14, -1):
        if time.time() > deadline - 10:
            break
        found = False
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            if trial < 5:
                # Structured initialization
                U, V, W = structured_init(target_rank)
            elif trial < 10:
                # Random integer init
                U = np.random.randint(-2, 3, (n * m, target_rank)).astype(float)
                V = np.random.randint(-2, 3, (m * p, target_rank)).astype(float)
                W = np.random.randint(-2, 3, (p * n, target_rank)).astype(float)
            else:
                # Random with some structure
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    U[i * m + j, r] = random.choice([-1, 1])
                    V[j * p + k, r] = random.choice([-1, 1])
                    W[k * n + i, r] = random.choice([-1, 1])
            
            # Coordinate descent with more iterations
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                a, b, c = random.choice(target_indices)
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(target_rank))
                err = 1.0 - s
                if abs(err) < 1e-12:
                    continue
                r = random.randint(0, target_rank - 1)
                d = V[b, r] * W[c, r]
                if abs(d) > 1e-8:
                    U[a, r] += err / d
                elif abs(U[a, r] * W[c, r]) > 1e-8:
                    V[b, r] += err / (U[a, r] * W[c, r])
                elif abs(U[a, r] * V[b, r]) > 1e-8:
                    W[c, r] += err / (U[a, r] * V[b, r])
            
            if verify(U, V, W, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
