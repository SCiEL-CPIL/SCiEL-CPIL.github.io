import numpy as np

n, m, p = 3, 4, 8
rank = 1
U = np.ones((n*m, rank))
V = np.ones((m*p, rank))
W = np.ones((p*n, rank))

U_3d = U.reshape(n, m, rank)
V_3d = V.reshape(m, p, rank)
W_3d = W.reshape(p, n, rank)

T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
print(T_recon)  # Should be all 1s if rank 1 works
