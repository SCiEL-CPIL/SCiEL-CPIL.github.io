<<<<<<< SEARCH
    # ---- Compression: joint U∧V∧W null-space elimination (the real refinement) ----
    def compress(Uc, Vc, Wc, tol=1e-9, max_iters=8):
        for _ in range(max_iters):
            if time.time() - _START > _DEADLINE - 15:
                break
            if len(Uc) <= 1:
                break
            Ua = np.array(Uc, dtype=float).T
            Va = np.array(Vc, dtype=float).T
            Wa = np.array(Wc, dtype=float).T
            U_s, _, U_Vt = np.linalg.svd(Ua, full_matrices=True)
            V_s, _, V_Vt = np.linalg.svd(Va, full_matrices=True)
            W_s, _, W_Vt = np.linalg.svd(Wa, full_matrices=True)
            rU = int(np.sum(U_s > tol))
            rV = int(np.sum(V_s > tol))
            rW = int(np.sum(W_s > tol))
            N_U = U_Vt[rU:]
            N_V = V_Vt[rV:]
            N_W = W_Vt[rW:]
            joint = []
            for a in N_U:
                if np.allclose(a, 0.0, atol=1e-12):
                    continue
                if (len(N_V) == 0 or np.allclose(N_V @ a, 0.0, atol=1e-10)) and \
                   (len(N_W) == 0 or np.allclose(N_W @ a, 0.0, atol=1e-10)):
                    joint.append(a)
            if not joint:
                break
            J = np.array(joint)
            # keep the top-r components (r = joint rank) -> provably preserves the
            # joint null space, i.e. the decomposition, while removing redundancy.
            J_s, _, J_Vt = np.linalg.svd(J, full_matrices=False)
            rJ = int(np.sum(J_s > tol))
            keep = J_Vt[:rJ]
            # drop the index whose coefficient is largest in the strongest kept vector
            drop_idx = int(np.argmax(np.abs(keep[0])))
            if abs(keep[0, drop_idx]) < 1e-12:
                break
            new_U = [Uc[i] for i in range(len(Uc)) if i != drop_idx]
            new_V = [Vc[i] for i in range(len(Vc)) if i != drop_idx]
            new_W = [Wc[i] for i in range(len(Wc)) if i != drop_idx]
            if verify(new_U, new_V, new_W):
                Uc, Vc, Wc = new_U, new_V, new_W
            else:
                break
        return Uc, Vc, Wc
=======
    # ---- Compression: tensor flattening + SVD-based redundancy removal ----
    def compress(Uc, Vc, Wc, tol=1e-9, max_iters=20):
        """
        View the multiplication tensor as a linear map from R^{n*m} x R^{m*p} -> R^{p*n}.
        Each rank-1 term r contributes: T_r(a,b) = (a^T u_r)(b^T v_r) w_r
        The full tensor T = sum_r u_r (v_r^T) w_r^T (in appropriate flattening).
        
        We build the "flattening" matrix M where M[:, r] encodes the joint effect,
        then use SVD to find directions in the rank-index space that are redundant.
        """
        for _ in range(max_iters):
            if time.time() - _START > _DEADLINE - 15:
                break
            if len(Uc) <= 1:
                break
            
            R = len(Uc)
            Ua = np.array(Uc, dtype=float).T  # (n*m, R)
            Va = np.array(Vc, dtype=float).T  # (m*p, R)
            Wa = np.array(Wc, dtype=float).T  # (p*n, R)
            
            # Build the "joint" matrix: for each pair of indices (u_idx, v_idx, w_idx),
            # the contribution is Ua[u_idx, r] * Va[v_idx, r] * Wa[w_idx, r]
            # The tensor T[i,k,j] = sum_r Ua[i*m+k, r] * Va[k*p+j, r] * Wa[j*n+i, r]
            # 
            # Flattening approach: consider the linear map from "input space" to "output space"
            # For a fixed flattening, we can write T as a matrix of size (n*p) x (m) or similar.
            # 
            # Better: use the "slice" approach. The tensor T has slices T_{:,k,:} for each k.
            # T_{:,k,:}[i,j] = sum_r Ua[i*m+k, r] * Va[k*p+j, r] * Wa[j*n+i, r]
            #
            # Key insight: if there exists a vector c in R^R such that
            #   c^T Ua = 0, c^T Va = 0, c^T Wa = 0 (all zero vectors)
            # then removing the component along c doesn't change T.
            # But that's the old approach.
            #
            # New approach: find c such that the "effective" contribution is zero.
            # The contribution of direction c is:
            #   delta_T[i,k,j] = sum_r c[r] * Ua[i*m+k, r] * Va[k*p+j, r] * Wa[j*n+i, r]
            # We want delta_T = 0 for all (i,k,j).
            #
            # This is a nonlinear constraint. Instead, we can look for c such that
            # the "weighted" vectors are simultaneously in the null spaces.
            #
            # Practical approach: iterate, each time finding the direction c that
            # minimizes ||delta_T||_F subject to ||c||=1, then if it's small enough,
            # project it out.
            
            # Simpler but effective: find c in the null space of Ua, Va, Wa simultaneously
            # by solving a least-squares problem.
            # Stack: minimize ||Ua^T c||^2 + ||Va^T c||^2 + ||Wa^T c||^2  s.t. ||c||=1
            # This is equivalent to finding the smallest eigenvector of
            # Ua@Ua^T + Va@Va^T + Wa@Wa^T
            
            S = Ua @ Ua.T + Va @ Va.T + Wa @ Wa.T
            eigvals, eigvecs = np.linalg.eigh(S)
            
            # Try removing components along the smallest eigenvectors
            removed_any = False
            for ei in range(min(5, R)):
                if time.time() - _START > _DEADLINE - 15:
                    break
                c = eigvecs[:, ei]
                # Check if this direction actually makes the tensor change small
                # delta_T[i,k,j] = sum_r c[r] * Ua[i*m+k, r] * Va[k*p+j, r] * Wa[j*n+i, r]
                # Approximate check: is c in approximate null space of each?
                nu = np.linalg.norm(Ua.T @ c)
                nv = np.linalg.norm(Va.T @ c)
                nw = np.linalg.norm(Wa.T @ c)
                
                if nu < tol and nv < tol and nw < tol:
                    # This is a true null direction - remove it
                    drop_idx = int(np.argmax(np.abs(c)))
                    if abs(c[drop_idx]) < 1e-12:
                        continue
                    new_U = [Uc[i] for i in range(R) if i != drop_idx]
                    new_V = [Vc[i] for i in range(R) if i != drop_idx]
                    new_W = [Wc[i] for i in range(R) if i != drop_idx]
                    if verify(new_U, new_V, new_W):
                        Uc, Vc, Wc = new_U, new_V, new_W
                        removed_any = True
                        break  # restart iteration with new matrices
                elif nu < 1e-6 and nv < 1e-6 and nw < 1e-6:
                    # Near-null: try removing and verify
                    drop_idx = int(np.argmax(np.abs(c)))
                    if abs(c[drop_idx]) < 1e-12:
                        continue
                    new_U = [Uc[i] for i in range(R) if i != drop_idx]
                    new_V = [Vc[i] for i in range(R) if i != drop_idx]
                    new_W = [Wc[i] for i in range(R) if i != drop_idx]
                    if verify(new_U, new_V, new_W, atol=1e-5):
                        Uc, Vc, Wc = new_U, new_V, new_W
                        removed_any = True
                        break
            
            # Also try: find pairs of terms that can be combined
            # If c1 and c2 are such that Ua.T@c1 = Ua.T@c2, Va.T@c1 = Va.T@c2, Wa.T*c1 = Wa.T*c2
            # then they're redundant. More practically: if two terms have proportional U, V, W.
            # Check for proportional pairs
            for i in range(len(Uc)):
                if time.time() - _START > _DEADLINE - 15:
                    break
                for j in range(i + 1, len(Uc)):
                    # Check if Uc[i] and Uc[j] are proportional
                    ui = Uc[i]
                    uj = Uc[j]
                    ni = np.linalg.norm(ui)
                    nj = np.linalg.norm(uj)
                    if ni < 1e-12 or nj < 1e-12:
                        continue
                    # Check U proportionality
                    cos_u = np.dot(ui, uj) / (ni * nj)
                    if abs(cos_u) < 0.999:
                        continue
                    # Check V
                    vi = Vc[i]
                    vj = Vc[j]
                    nvi = np.linalg.norm(vi)
                    nvj = np.linalg.norm(vj)
                    if nvi < 1e-12 or nvj < 1e-12:
                        continue
                    cos_v = np.dot(vi, vj) / (nvi * nvj)
                    if abs(cos_v) < 0.999:
                        continue
                    # Check W
                    wi = Wc[i]
                    wj = Wc[j]
                    nwi = np.linalg.norm(wi)
                    nwj = np.linalg.norm(wj)
                    if nwi < 1e-12 or nwj < 1e-12:
                        continue
                    cos_w = np.dot(wi, wj) / (nwi * nwj)
                    if abs(cos_w) < 0.999:
                        continue
                    
                    # All three are proportional - check if they cancel
                    # If U[i] = a*U[j], V[i] = b*V[j], W[i] = c*W[j]
                    # then term i = a*b*c * term j
                    # If a*b*c ≈ 0, we can remove i
                    # Estimate a, b, c
                    if np.dot(ui, uj) > 0:
                        a = np.dot(ui, uj) / (ni * nj)
                    else:
                        a = -np.dot(ui, uj) / (ni * nj)
                    if np.dot(vi, vj) > 0:
                        b = np.dot(vi, vj) / (nvi * nvj)
                    else:
                        b = -np.dot(vi, vj) / (nvi * nvj)
                    if np.dot(wi, wj) > 0:
                        c = np.dot(wi, wj) / (nwi * nwj)
                    else:
                        c = -np.dot(wi, wj) / (nwi * nwj)
                    
                    abc = a * b * c
                    if abs(abc) < 1e-6:
                        # Term i is negligible compared to term j
                        new_U = [Uc[k] for k in range(len(Uc)) if k != i]
                        new_V = [Vc[k] for k in range(len(Vc)) if k != i]
                        new_W = [Wc[k] for k in range(len(Wc)) if k != i]
                        if verify(new_U, new_V, new_W):
                            Uc, Vc, Wc = new_U, new_V, new_W
                            removed_any = True
                            break
                if removed_any:
                    break
            
            if not removed_any:
                break
        
        return Uc, Vc, Wc
>>>>>>> REPLACE

