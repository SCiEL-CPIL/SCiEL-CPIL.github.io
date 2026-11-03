import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_np(U, V, W, rank):
        """Fast vectorized verification using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Compute D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        D = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        # Check against expected tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(D[a, b, c] - 1.0) > 1e-9:
                        return False
        # Check that all other entries are 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (i == i2 and j == j2 and k == k2):
                        if abs(D[a, b, c]) > 1e-9:
                            return False
        return True
    
    def try_merge_pairs(schoolbook_U, schoolbook_V, schoolbook_W):
        """
        Try to merge pairs of schoolbook terms into single terms.
        A schoolbook term for (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
        To merge two terms (i1,j1,k1) and (i2,j2,k2) into one term, we need:
        - The merged term to contribute to both (i1,j1,k1) and (i2,j2,k2)
        - And cancel out any unwanted contributions
        
        For a term to contribute to position (i,j,k), it needs:
        U[i*m+j] * V[j*p+k] * W[k*n+i] to be nonzero.
        
        Let's try: for pairs where they share some indices, see if we can create
        a term that covers both with appropriate signs/values.
        """
        best_rank = n * m * p
        best_result = {"U": schoolbook_U, "V": schoolbook_V, "W": schoolbook_W}
        
        # Build list of all (i,j,k) positions
        positions = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Try all pairs of positions to see if they can be merged
        # A merge is possible if we can find u, v, w values such that:
        # For position (i1,j1,k1): u1*v1*w1 = 1 (contribution should be 1)
        # For position (i2,j2,k2): u2*v2*w2 = 1 (contribution should be 1)  
        # For all other positions: contribution = 0
        
        # Actually, let's think differently. Start with schoolbook and try to
        # replace pairs of terms with a single term that has half-integer values.
        
        # For two terms t1=(i1,j1,k1) and t2=(i2,j2,k2), can we find a single
        # term that replaces both? The new term would need:
        # U[i1*m+j1]*V[j1*p+k1]*W[k1*n+i1] = 1
        # U[i2*m+j2]*V[j2*p+k2]*W[k2*n+i2] = 1
        # And for all other (i,j,k): U[i*m+j]*V[j*p+k]*W[k*n+i] = 0
        
        # This is generally not possible with a single term unless the positions
        # are "compatible" in a specific way.
        
        # Alternative: use 2 terms to replace 3 schoolbook terms (net gain of 1)
        # Or find algebraic identities.
        
        # Let's try a different approach: use the known structure of matrix multiplication
        # and try to find decompositions using block methods.
        
        # For 3x3 * 3x4, we can think of it as:
        # [A1 A2 A3] * [B1 B2 B3 B4] = [A1*B1+A2*B2+A3*B3, ...]
        # where A1,A2,A3 are 3x3 blocks and B1..B4 are 3x1 columns.
        
        # Actually, let's try a more systematic approach:
        # Search over small sets of "basis" terms with values in {-1, -0.5, 0, 0.5, 1}
        # and see if we can cover all 36 positions.
        
        # Given time constraints, let's try a focused search:
        # 1. Keep most schoolbook terms
        # 2. Try to merge specific pairs that share structure
        
        # For example, if two terms share the same i but different j,k:
        # Term1: (i, j1, k1) -> U[i*m+j1]=1, V[j1*p+k1]=1, W[k1*n+i]=1
        # Term2: (i, j2, k2) -> U[i*m+j2]=1, V[j2*p+k2]=1, W[k2*n+i]=1
        # These can't easily be merged into one term.
        
        # Let's try a randomized local search starting from schoolbook:
        # Pick a random pair of terms, try to replace them with 1 or 2 new terms
        # that have half-integer values and still cover the required positions.
        
        U_curr = [row[:] for row in schoolbook_U]
        V_curr = [row[:] for row in schoolbook_V]
        W_curr = [row[:] for row in schoolbook_W]
        curr_rank = n * m * p
        
        vals = [-1, -0.5, 0, 0.5, 1]
        
        max_attempts = 5000
        attempt = 0
        while attempt < max_attempts and time.time() < deadline:
            attempt += 1
            
            # Pick two random distinct terms to try to merge
            t1 = random.randint(0, curr_rank - 1)
            t2 = random.randint(0, curr_rank - 1)
            if t1 == t2:
                continue
            
            # Try to replace terms t1 and t2 with a single new term
            # The new term needs to satisfy the constraints for both positions
            
            # Get the positions covered by t1 and t2
            # Term t contributes to position (i,j,k) if U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] != 0
            # In schoolbook, each term covers exactly one position.
            
            # Find which positions t1 and t2 cover
            pos1 = None
            pos2 = None
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if abs(U_curr[a][t1]) > 0 and abs(V_curr[b][t1]) > 0 and abs(W_curr[c][t1]) > 0:
                            if U_curr[a][t1] * V_curr[b][t1] * W_curr[c][t1] != 0:
                                pos1 = (i, j, k)
                        if abs(U_curr[a][t2]) > 0 and abs(V_curr[b][t2]) > 0 and abs(W_curr[c][t2]) > 0:
                            if U_curr[a][t2] * V_curr[b][t2] * W_curr[c][t2] != 0:
                                pos2 = (i, j, k)
            
            if pos1 is None or pos2 is None:
                continue
            
            i1, j1, k1 = pos1
            i2, j2, k2 = pos2
            
            # For a single term to cover both positions:
            # U[i1*m+j1]*V[j1*p+k1]*W[k1*n+i1] = 1
            # U[i2*m+j2]*V[j2*p+k2]*W[k2*n+i2] = 1
            # And for all other positions, the product should be 0.
            
            # This is only possible if the two positions are "compatible"
            # Let's try: set the new term's values to cover both positions
            # and check if it accidentally covers other positions.
            
            new_U = [0] * (n * m)
            new_V = [0] * (m * p)
            new_W = [0] * (p * n)
            
            # We need to find u_a, v_b, w_c values such that:
            # u_{i1*m+j1} * v_{j1*p+k1} * w_{k1*n+i1} = 1
            # u_{i2*m+j2} * v_{j2*p+k2} * w_{k2*n+i2} = 1
            # and u_a * v_b * w_c = 0 for all other (a,b,c) in the tensor support
            
            # Simplest: try to set values so that both products are 1
            # and see if we can make all other products 0.
            
            # This is generally hard. Let's skip this approach and try something else.
            
            # Instead, let's try: can we find 2 terms that replace 3 schoolbook terms?
            # This would reduce rank by 1.
            
            # Pick 3 random terms
            t3 = random.randint(0, curr_rank - 1)
            if t3 == t1 or t3 == t2:
                continue
            
            pos3 = None
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if U_curr[a][t3] * V_curr[b][t3] * W_curr[c][t3] != 0:
                            pos3 = (i, j, k)
            
            if pos3 is None:
                continue
            
            # Now try to find 2 terms that cover pos1, pos2, pos3
            # This is a small search problem. Let's try random values.
            
            found = False
            for trial in range(100):
                if time.time() > deadline:
                    break
                
                # Randomly assign values to the 2 new terms
                U_new = [[0] * 2 for _ in range(n * m)]
                V_new = [[0] * 2 for _ in range(m * p)]
                W_new = [[0] * 2 for _ in range(p * n)]
                
                for t in range(2):
                    for a in range(n * m):
                        U_new[a][t] = random.choice(vals)
                    for b in range(m * p):
                        V_new[b][t] = random.choice(vals)
                    for c in range(p * n):
                        W_new[c][t] = random.choice(vals)
                
                # Check if these 2 terms can cover the 3 positions
                # and don't create unwanted nonzero entries
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Compute contribution from the 2 new terms
                            contrib = 0
                            for t in range(2):
                                contrib += U_new[a][t] * V_new[b][t] * W_new[c][t]
                            
                            # This position should be covered by the new terms
                            # if it's one of pos1, pos2, pos3
                            if (i, j, k) in [pos1, pos2, pos3]:
                                if abs(contrib - 1.0) > 1e-9:
                                    ok = False
                                    break
                            else:
                                # Other positions should have 0 contribution from new terms
                                # (they'll be covered by remaining schoolbook terms)
                                if abs(contrib) > 1e-9:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    # Success! Replace terms t1, t2, t3 with the 2 new terms
                    # Remove the 3 old terms and add the 2 new ones
                    # Net change: -1 in rank
                    
                    # Build new U, V, W without the 3 old terms
                    U_new_full = []
                    V_new_full = []
                    W_new_full = []
                    
                    for a in range(n * m):
                        row = []
                        for t in range(curr_rank):
                            if t in [t1, t2, t3]:
                                # This term is being removed
                                continue
                            else:
                                row.append(U_curr[a][t])
                        # Add the 2 new terms
                        row.extend([U_new[a][0], U_new[a][1]])
                        U_new_full.append(row)
                    
                    for b in range(m * p):
                        row = []
                        for t in range(curr_rank):
                            if t in [t1, t2, t3]:
                                continue
                            else:
                                row.append(V_curr[b][t])
                        row.extend([V_new[b][0], V_new[b][1]])
                        V_new_full.append(row)
                    
                    for c in range(p * n):
                        row = []
                        for t in range(curr_rank):
                            if t in [t1, t2, t3]:
                                continue
                            else:
                                row.append(W_curr[c][t])
                        row.extend([W_new[c][0], W_new[c][1]])
                        W_new_full.append(row)
                    
                    new_rank = curr_rank - 1
                    
                    # Verify the full decomposition
                    if verify_np(U_new_full, V_new_full, W_new_full, new_rank):
                        U_curr = U_new_full
                        V_curr = V_new_full
                        W_curr = W_new_full
                        curr_rank = new_rank
                        if curr_rank < best_rank:
                            best_rank = curr_rank
                            best_result = {"U": U_curr, "V": V_curr, "W": W_curr}
                        found = True
                        break
            
            if found:
                continue
        
        return best_result
    
    # First get schoolbook
    sb = schoolbook()
    
    # Try to improve
    result = try_merge_pairs(sb["U"], sb["V"], sb["W"])
    
    return result
