                # Update U
                Z_u = np.einsum('vr,wr->vwr', V, W).reshape(m * p * p * n, rank)
                for u in range(n * m):
                    target_u = target_flat[u].flatten()
                    U[u, :], _, _, _ = np.linalg.lstsq(Z_u, target_u, rcond=None)
                        
                # Update V
                Z_v = np.einsum('ur,wr->uwr', U, W).reshape(n * m * p * n, rank)
                for v in range(m * p):
                    target_v = target_flat[:, v, :].flatten()
                    V[v, :], _, _, _ = np.linalg.lstsq(Z_v, target_v, rcond=None)
                        
                # Update W
                Z_w = np.einsum('ur,vr->uvr', U, V).reshape(n * m * m * p, rank)
                for w in range(p * n):
                    target_w = target_flat[:, :, w].flatten()
                    W[w, :], _, _, _ = np.linalg.lstsq(Z_w, target_w, rcond=None)
