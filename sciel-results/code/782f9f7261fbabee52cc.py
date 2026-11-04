import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    deadline = time.time() + 110

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a better rank
    # We know schoolbook is 45.
    # We aim for 41-44.

    for R in [44, 43, 42, 41, 40, 39, 38]:
        if time.time() > deadline:
            break

        for attempt in range(50):
            if time.time() > deadline:
                break

            # Initialize with small integers/half-integers
            # Use a sparse initialization to help convergence
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, R))

            # Reshape for easier index handling
            # U: (n, m, R)
            # V: (m, p, R)
            # W: (p, n, R)
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)

            # Coordinate descent
            for step in range(20):
                if time.time() > deadline:
                    break

                # Update U
                # U_{ijr} = sum_k (1 - C_{ijk}) V_{jkr} W_{kir} / sum_k V_{jkr}^2 W_{kir}^2
                # C_{ijk} = sum_{r' != r} U_{ijr'} V_{jkr'} W_{kir'}

                # Compute C for all i,j,k,r
                # C_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
                # This is the current product.
                # We need C excluding r.
                # Product_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}

                # Let's compute the full product P_{ijk}
                # P_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
                # Using einsum:
                # U: (n, m, R), V: (m, p, R), W: (p, n, R)
                # P_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                # Note: W index is k, i. So W_{k,i,r}.
                # W_3d is (p, n, R) which is (k, i, r). Correct.

                P = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

                # For each r, we need to update U_{ijr}.
                # Optimal U_{ijr} = (sum_k (1 - (P_{ijk} - U_{ijr}V_{jkr}W_{kir})) V_{jkr} W_{kir}) / (sum_k V_{jkr}^2 W_{kir}^2)
                # = (sum_k (1 - P_{ijk} + U_{ijr}V_{jkr}W_{kir}) V_{jkr} W_{kir}) / Denom
                # = (sum_k (1 - P_{ijk}) V_{jkr} W_{kir} + U_{ijr} sum_k V_{jkr}^2 W_{kir}^2) / Denom
                # = (Numerator_base_{ijr} + U_{ijr} Denom_{ijr}) / Denom_{ijr}
                # = Numerator_base_{ijr} / Denom_{ijr} + U_{ijr}
                # Wait, this is wrong. The term U_{ijr} is inside the sum for C.
                # C_{ijk}^{(r)} = P_{ijk} - U_{ijr} V_{jkr} W_{kir}
                # 1 - C = 1 - P + U_{ijr} V_{jkr} W_{kir}
                # Num = sum_k (1 - P_{ijk}) V_{jkr} W_{kir} + U_{ijr} sum_k V_{jkr}^2 W_{kir}^2
                # Denom = sum_k V_{jkr}^2 W_{kir}^2
                # U_new = (Num_base + U_old * Denom) / Denom = Num_base / Denom + U_old
                # This implies U_new = U_old + Num_base / Denom.
                # This is a gradient step? No, it's the exact solution for a linear system?
                # Let's re-derive.
                # Minimize sum_k (U V W + C_other - 1)^2
                # d/du (u^2 D + 2 u N + ...) = 2 u D + 2 N = 0 => u = -N/D
                # Here N = sum_k (C_other - 1) V W
                # C_other = P - u V W
                # N = sum_k (P - u V W - 1) V W = sum_k (P-1) V W - u sum_k V^2 W^2
                # u D + sum_k (P-1) V W - u D = 0?
                # Let's just use the formula:
                # u_opt = sum_k (1 - C_other) V W / sum_k V^2 W^2
                # C_other = P - u V W
                # 1 - C_other = 1 - P + u V W
                # u_opt = sum_k (1 - P) V W / D + u sum_k V^2 W^2 / D
                # u_opt = Num_base / D + u
                # So u_new = u + Num_base / D.

                # Calculate D_{ijr} = sum_k V_{jkr}^2 W_{kir}^2
                # V_3d: (m, p, R) -> V_{jkr}
                # W_3d: (p, n, R) -> W_{kir}
                # D_{jkr} = V_sq[j,k,r] * W_sq[k,i,r]? No, sum over k.
                # D_{jr} = sum_k V_{jkr}^2 W_{kir}^2 ? No, W depends on i.
                # W_{kir}. So D depends on i, j, r.
                # D_{ijr} = sum_k V_{jkr}^2 W_{kir}^2

                # Num_base_{ijr} = sum_k (1 - P_{ijk}) V_{jkr} W_{kir}

                # Let's compute these using einsum.
                # V_3d: (m, p, R)
                # W_3d: (p, n, R)
                # P: (n, m, p)

                # D_{ijr} = sum_k V_{jkr}^2 W_{kir}^2
                # V_sq: (m, p, R)
                # W_sq: (p, n, R)
                # D = np.einsum('jkr,kir->ijr', V_sq, W_sq) ? No, indices don't match for i.
                # V has no i. W has i.
                # D_{ijr} = sum_k V_{jkr}^2 W_{kir}^2
                # This is a tensor of shape (n, m, R).
                # V_sq[j, k, r] * W_sq[k, i, r]
                # Sum over k.
                # D = np.einsum('jkr,kir->ijr', V_sq, W_sq)

                # Num_base_{ijr} = sum_k (1 - P_{ijk}) V_{jkr} W_{kir}
                # (1 - P_{ijk}) is (n, m, p)
                # V_{jkr} is (m, p, R)
                # W_{kir} is (p, n, R)
                # Term = (1 - P)_{ijk} * V_{jkr} * W_{kir}
                # Sum over k.
                # Result shape: (n, m, R)
                # Num_base = np.einsum('ijk,jkr,kir->ijr', 1 - P, V_3d, W_3d)

                # Update U_3d
                # U_new = U + Num_base / D
                # Avoid division by zero
                V_sq = V_3d**2
                W_sq = W_3d**2
                D = np.einsum('jkr,kir->ijr', V_sq, W_sq)
                Num_base = np.einsum('ijk,jkr,kir->ijr', 1 - P, V_3d, W_3d)
                D_safe = np.where(D > 1e-10, D, 1.0)
                U_3d += Num_base / D_safe

                # Similarly for V and W
                # V_{jkr} update
                # C_other for V: P - U V W
                # D_{jkr} = sum_i U_{ijr}^2 W_{kir}^2
                # Num_base_{jkr} = sum_i (1 - P_{ijk}) U_{ijr} W_{kir}

                U_sq = U_3d**2
                W_sq = W_3d**2
                D_v = np.einsum('ijr,kir->jkr', U_sq, W_sq)
                Num_base_v = np.einsum('ijk,ijr,kir->jkr', 1 - P, U_3d, W_3d)
                D_v_safe = np.where(D_v > 1e-10, D_v, 1.0)
                V_3d += Num_base_v / D_v_safe

                # W_{kir} update
                # D_{kir} = sum_j U_{ijr}^2 V_{jkr}^2
                # Num_base_{kir} = sum_j (1 - P_{ijk}) U_{ijr} V_{jkr}

                U_sq = U_3d**2
                V_sq = V_3d**2
                D_w = np.einsum('ijr,jkr->kir', U_sq, V_sq)
                Num_base_w = np.einsum('ijk,ijr,jkr->kir', 1 - P, U_3d, V_3d)
                D_w_safe = np.where(D_w > 1e-10, D_w, 1.0)
                W_3d += Num_base_w / D_w_safe

                # Recompute P
                P = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

                # Check error
                if np.max(np.abs(P - 1)) < 1e-6:
                    break

            if np.max(np.abs(P - 1)) < 1e-6:
                # Check if entries are close to integers or half-integers
                U_2d = U_3d.reshape(nm, R)
                V_2d = V_3d.reshape(mp, R)
                W_2d = W_3d.reshape(pn, R)

                def is_valid_matrix(M):
                    # Check if all entries are close to half-integers
                    doubled = M * 2
                    rounded = np.round(doubled)
                    if np.max(np.abs(doubled - rounded)) < 1e-4:
                        return True
                    return False

                if is_valid_matrix(U_2d) and is_valid_matrix(V_2d) and is_valid_matrix(W_2d):
                    # Round and return
                    U_final = np.round(U_2d * 2) / 2
                    V_final = np.round(V_2d * 2) / 2
                    W_final = np.round(W_2d * 2) / 2

                    # Final verification
                    U_f_3d = U_final.reshape(n, m, R)
                    V_f_3d = V_final.reshape(m, p, R)
                    W_f_3d = W_final.reshape(p, n, R)
                    P_final = np.einsum('ijr,jkr,kir->ijk', U_f_3d, V_f_3d, W_f_3d)

                    if np.max(np.abs(P_final - 1)) < 1e-9:
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    return schoolbook()
