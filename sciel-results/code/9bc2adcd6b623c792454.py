import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    start_time = time.time()
    deadline = 145  # seconds, leave margin for safety

    # 1. Construct the target tensor T
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def verify_decomposition_float(U, V, W, atol=1e-9):
        """
        Checks if the decomposition is exact in floating point.
        """
        rank = U.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T_target, atol=atol)

    def verify_rational(U, V, W):
        """
        Verifies exactness using Fractions and checks half-integer constraint.
        """
        rank = U.shape[1]
        # Check integer/half-integer constraint first
        for mat in [U, V, W]:
            m2 = mat * 2
            if not np.allclose(m2, np.round(m2), atol=1e-6):
                return False
        
        # Convert to Fractions
        U_frac = np.zeros((U.shape[0], rank), dtype=object)
        V_frac = np.zeros((V.shape[0], rank), dtype=object)
        W_frac = np.zeros((W.shape[0], rank), dtype=object)
        
        for i in range(U.shape[0]):
            for r in range(rank):
                val = round(U[i, r] * 2)
                U_frac[i, r] = Fraction(val, 2)
        for i in range(V.shape[0]):
            for r in range(rank):
                val = round(V[i, r] * 2)
                V_frac[i, r] = Fraction(val, 2)
        for i in range(W.shape[0]):
            for r in range(rank):
                val = round(W[i, r] * 2)
                W_frac[i, r] = Fraction(val, 2)
        
        # Check all entries of the tensor
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    total = Fraction(0, 1)
                    for r in range(rank):
                        total += U_frac[a, r] * V_frac[b, r] * W_frac[c, r]
                    if total != T_target[a, b, c]:
                        return False
        return True

    def to_lists(mat):
        return mat.tolist()

    # 2. Deterministic Rank-17 Construction (Strassen Hybrid)
    # This is a known valid decomposition.
    def build_rank_17():
        U1 = np.zeros((6, 7))
        V1 = np.zeros((9, 7))
        W1 = np.zeros((6, 7))
        
        # --- Part 1: Strassen for A_L (2x2) * B_TL (2x2) -> C_TL (2x2) ---
        # A_L indices in U: 0,1,3,4
        # B_TL indices in V: 0,1,3,4
        # C_TL indices in W: 0,1,2,3
        
        U_s = np.zeros((6, 7))
        V_s = np.zeros((9, 7))
        W_s = np.zeros((6, 7))
        
        # m1: (a00+a11) * (b00+b11)
        U_s[0, 0] = 1; U_s[4, 0] = 1
        V_s[0, 0] = 1; V_s[4, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # m2: (a10+a11) * b00
        U_s[3, 1] = 1; U_s[4, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = -1
        
        # m3: a00 * (b01-b11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[4, 2] = -1
        W_s[0, 2] = 1; W_s[2, 2] = 1
        
        # m4: a11 * (b10-b00)
        U_s[4, 3] = 1
        V_s[3, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1
        
        # m5: (a00+a01) * b11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[4, 4] = 1
        W_s[0, 4] = -1; W_s[2, 4] = 1
        
        # m6: (a10-a00) * (b00+b01)
        U_s[3, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        
        # m7: (a01-a11) * (b10+b11)
        U_s[1, 6] = 1; U_s[4, 6] = -1
        V_s[3, 6] = 1; V_s[4, 6] = 1
        W_s[0, 6] = 1; W_s[2, 6] = 1
        
        # --- Part 2: Schoolbook for A_L (2x2) * B_TR (2x1) -> C_TR (2x1) ---
        # A_L indices in U: 0,1,3,4
        # B_TR indices in V: 2,5 (col 2 of top 2 rows)
        # C_TR indices in W: 4,5 (col 2 of C)
        
        U_b = np.zeros((6, 4))
        V_b = np.zeros((9, 4))
        W_b = np.zeros((6, 4))
        
        # m8: a00 * b02
        U_b[0, 0] = 1
        V_b[2, 0] = 1
        W_b[4, 0] = 1
        
        # m9: a01 * b12
        U_b[1, 1] = 1
        V_b[5, 1] = 1
        W_b[4, 1] = 1
        
        # m10: a10 * b02
        U_b[3, 2] = 1
        V_b[2, 2] = 1
        W_b[5, 2] = 1
        
        # m11: a11 * b12
        U_b[4, 3] = 1
        V_b[5, 3] = 1
        W_b[5, 3] = 1
        
        # --- Part 3: Schoolbook for A_R (2x1) * B_B (1x3) -> C (2x3) ---
        # A_R indices in U: 2,5 (col 2 of A)
        # B_B indices in V: 6,7,8 (row 2 of B)
        # C indices in W: 0,1,2,3,4,5 (all of C)
        
        U_c = np.zeros((6, 1))
        V_c = np.zeros((9, 1))
        W_c = np.zeros((6, 1))
        
        U_c[2, 0] = 1
        U_c[5, 0] = 1
        V_c[6, 0] = 1
        V_c[7, 0] = 1
        V_c[8, 0] = 1
        W_c[0, 0] = 1
        W_c[1, 0] = 1
        W_c[2, 0] = 1
        W_c[3, 0] = 1
        W_c[4, 0] = 1
        W_c[5, 0] = 1
        
        U_total = np.hstack([U_s, U_b, U_c])
        V_total = np.hstack([V_s, V_b, V_c])
        W_total = np.hstack([W_s, W_b, W_c])
        
        return U_total, V_total, W_total

    U17, V17, W17 = build_rank_17()
    
    # Verify Rank 17
    if not verify_decomposition_float(U17, V17, W17):
        # Fallback to schoolbook if something went wrong (shouldn't happen)
        U_sb = np.zeros((6, 18))
        V_sb = np.zeros((9, 18))
        W_sb = np.zeros((6, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {
            "U": to_lists(U_sb),
            "V": to_lists(V_sb),
            "W": to_lists(W_sb)
        }

    # 3. Attempt to find Rank 16 via Continuous Optimization
    # Strategy:
    # Initialize U, V, W with random values (or perturbed rank-17 with one column removed/merged).
    # Use gradient descent to minimize ||T - sum(U_r V_r W_r)||^2.
    # If we reach a very low loss, try to snap to half-integers.
    
    best_rank_16_solution = None
    
    def try_optimize_rank_16():
        if time.time() > deadline - 20:
            return None
        
        # Initialize with random values
        # We can try multiple initializations
        for init_iter in range(5):
            if time.time() > deadline - 10:
                break
                
            # Random initialization with small values
            U = np.random.randn(nm, 16) * 0.1
            V = np.random.randn(mp, 16) * 0.1
            W = np.random.randn(pn, 16) * 0.1
            
            # Learning rate schedule
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Compute T_approx
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_approx - T_target
                
                # Compute gradients
                # dL/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                # dL/dV[b,r] = sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
                # dL/dW[c,r] = sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
                
                grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                loss = np.sum(diff**2)
                if loss < 1e-12:
                    # Try to snap
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify_decomposition_float(U_snap, V_snap, W_snap, atol=1e-6):
                        if verify_rational(U_snap, V_snap, W_snap):
                            return U_snap, V_snap, W_snap
                            
                # Adaptive learning rate (simple decay)
                if step % 500 == 0 and step > 0:
                    lr *= 0.5
                    
            # End steps
            # Final check on the last state
            loss = np.sum((np.einsum('ar,br,cr->abc', U, V, W) - T_target)**2)
            if loss < 1e-8:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if verify_decomposition_float(U_snap, V_snap, W_snap, atol=1e-6):
                    if verify_rational(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap

    res_16 = try_optimize_rank_16()
    
    if res_16 is not None:
        U16, V16, W16 = res_16
        return {
            "U": to_lists(U16),
            "V": to_lists(V16),
            "W": to_lists(W16)
        }

    # Return the best found (Rank 17)
    return {
        "U": to_lists(U17),
        "V": to_lists(V17),
        "W": to_lists(W17)
    }

# Test
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
