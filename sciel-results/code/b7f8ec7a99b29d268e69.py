import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

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

    def try_rank(rank):
        if time.time() > deadline - 15:
            return None

        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank

        # Precompute: for each (i,j,k), the dot product contributions
        # s[i,j,k] = sum_r U[(i*m+j)*rank+r] * V[(j*p+k)*rank+r] * W[(k*n+i)*rank+r]
        # loss = sum_{i,j,k} (s[i,j,k] - 1)^2

        # Precompute index slices
        u_idx = {}
        v_idx = {}
        w_idx = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    base_u = (i*m+j)*rank
                    base_v = (j*p+k)*rank
                    base_w = (k*n+i)*rank
                    u_idx[(i,j,k)] = np.arange(base_u, base_u+rank)
                    v_idx[(i,j,k)] = np.arange(base_v, base_v+rank)
                    w_idx[(i,j,k)] = np.arange(base_w, base_w+rank)

        u_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        def full_loss(U_f, V_f, W_f):
            """Compute full loss vectorized."""
            s_vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s_vals[i,j,k] = np.dot(
                            np.dot(U_f[u_idx[(i,j,k)]], V_f[v_idx[(i,j,k)]]),
                            W_f[w_idx[(i,j,k)]]
                        )
            return np.sum((s_vals - 1.0)**2)

        def incremental_loss_change(U_f, V_f, W_f, var_type, idx, old_val, new_val):
            """
            Compute the change in loss when changing one element.
            Returns delta_loss = new_loss - old_loss.
            """
            delta = new_val - old_val
            delta_loss = 0.0

            if var_type == 0:  # U
                ij = idx // rank
                r = idx % rank
                i = ij // m
                j = ij % m
                for k in range(p):
                    key = (i,j,k)
                    u_slice = U_f[u_idx[key]]
                    v_slice = V_f[v_idx[key]]
                    w_slice = W_f[w_idx[key]]
                    # s_old = dot(u_slice, v_slice*w_slice)
                    # u_new = u_old + delta at position r
                    # s_new = s_old + delta * v_slice[r] * w_slice[r]
                    s_old = np.dot(u_slice, v_slice * w_slice)
                    s_new = s_old + delta * v_slice[r] * w_slice[r]
                    delta_loss += (s_new - 1.0)**2 - (s_old - 1.0)**2

            elif var_type == 1:  # V
                jk = idx // rank
                r = idx % rank
                j = jk // p
                k = jk % p
                for i in range(n):
                    key = (i,j,k)
                    u_slice = U_f[u_idx[key]]
                    v_slice = V_f[v_idx[key]]
                    w_slice = W_f[w_idx[key]]
                    s_old = np.dot(u_slice, v_slice * w_slice)
                    s_new = s_old + delta * u_slice[r] * w_slice[r]
                    delta_loss += (s_new - 1.0)**2 - (s_old - 1.0)**2

            else:  # W
                ki = idx // rank
                r = idx % rank
                k = ki // n
                i = ki % n
                for j in range(m):
                    key = (i,j,k)
                    u_slice = U_f[u_idx[key]]
                    v_slice = V_f[v_idx[key]]
                    w_slice = W_f[w_idx[key]]
                    s_old = np.dot(u_slice, v_slice * w_slice)
                    s_new = s_old + delta * u_slice[r] * v_slice[r]
                    delta_loss += (s_new - 1.0)**2 - (s_old - 1.0)**2

            return delta_loss

        max_restarts = 8
        elapsed = time.time() - (deadline - 140)  # approximate
        remaining = deadline - 10 - time.time()
        per_restart = max(2.0, min(15.0, remaining / max(1, max_restarts)))

        for restart in range(max_restarts):
            if time.time() > deadline - 8:
                break

            U_flat = np.random.choice(u_vals, size=size_u)
            V_flat = np.random.choice(u_vals, size=size_v)
            W_flat = np.random.choice(u_vals, size=size_w)

            current_loss = full_loss(U_flat, V_flat, W_flat)
            best_loss = current_loss
            best_U = U_flat.copy()
            best_V = V_flat.copy()
            best_W = W_flat.copy()

            temp = 1.0
            cooling = 0.9995
            restart_deadline = time.time() + per_restart
            iters = 0

            while time.time() < restart_deadline:
                iters += 1
                if iters % 5000 == 0 and temp < 0.05:
                    break

                var_type = random.randint(0, 2)
                if var_type == 0:
                    idx = random.randint(0, size_u - 1)
                    old_val = U_flat[idx]
                    new_val = random.choice(u_vals)
                    if new_val == old_val:
                        continue
                    delta_loss = incremental_loss_change(U_flat, V_flat, W_flat, 0, idx, old_val, new_val)
                    new_loss = current_loss + delta_loss
                    if new_loss > current_loss:
                        if random.random() > np.exp((current_loss - new_loss) / max(temp, 0.01)):
                            continue
                    U_flat[idx] = new_val
                    current_loss = new_loss
                elif var_type == 1:
                    idx = random.randint(0, size_v - 1)
                    old_val = V_flat[idx]
                    new_val = random.choice(u_vals)
                    if new_val == old_val:
                        continue
                    delta_loss = incremental_loss_change(U_flat, V_flat, W_flat, 1, idx, old_val, new_val)
                    new_loss = current_loss + delta_loss
                    if new_loss > current_loss:
                        if random.random() > np.exp((current_loss - new_loss) / max(temp, 0.01)):
                            continue
                    V_flat[idx] = new_val
                    current_loss = new_loss
                else:
                    idx = random.randint(0, size_w - 1)
                    old_val = W_flat[idx]
                    new_val = random.choice(u_vals)
                    if new_val == old_val:
                        continue
                    delta_loss = incremental_loss_change(U_flat, V_flat, W_flat, 2, idx, old_val, new_val)
                    new_loss = current_loss + delta_loss
                    if new_loss > current_loss:
                        if random.random() > np.exp((current_loss - new_loss) / max(temp, 0.01)):
                            continue
                    W_flat[idx] = new_val
                    current_loss = new_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_U = U_flat.copy()
                    best_V = V_flat.copy()
                    best_W = W_flat.copy()

                    if best_loss < 1e-9:
                        U_list = [[best_U[(i*m+j)*rank + r] for r in range(rank)] for i in range(n) for j in range(m)]
                        V_list = [[best_V[(j*p+k)*rank + r] for r in range(rank)] for j in range(m) for k in range(p)]
                        W_list = [[best_W[(k*n+i)*rank + r] for r in range(rank)] for k in range(p) for i in range(n)]

                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = 0
                                    for r in range(rank):
                                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                                    if abs(s - 1.0) > 1e-9:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            return {"U": U_list, "V": V_list, "W": W_list}

                temp *= cooling

        return None

    for target_rank in range(27, 37):
        if time.time() > deadline - 10:
            break
        result = try_rank(target_rank)
        if result is not None:
            return result

    return schoolbook()
# EVOLVE-BLOCK-END
