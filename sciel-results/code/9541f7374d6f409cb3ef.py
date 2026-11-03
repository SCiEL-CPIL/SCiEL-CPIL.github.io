        # Simulated annealing to escape local minima.
        T_start = 0.05
        sa_steps = 5000
        for it in range(sa_steps):
            if _now() > deadline - 25:
                break
            T = max(T_start * (0.995 ** it), 1e-6)
            
            if np.random.rand() < 0.2:
                # 2-flip move
                idx1 = int(np.random.randint(N))
                idx2 = int(np.random.randint(N))
                if idx1 == idx2:
                    continue
                new_cur = cur.copy()
                new_cur[idx1] = -new_cur[idx1]
                new_cur[idx2] = -new_cur[idx2]
                new_ac = fftconvolve(new_cur, new_cur, mode='full')
                new_sum = np.sum(new_cur)
                nsq = new_sum ** 2
                if nsq < 1e-10:
                    continue
                new_score = float(np.max(np.abs(new_ac)) / nsq)
                
                d = new_score - cur_score
                if d < 0 or np.random.rand() < np.exp(-d / T):
                    cur = new_cur
                    cur_ac = new_ac
                    cur_sum = new_sum
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_s = cur.copy()
            else:
                # 1-flip move
                idx = int(np.random.randint(N))
                new_score = _max_ac_after_flip(cur, cur_ac, cur_sum, idx)
                if new_score == np.inf:
                    continue
                d = new_score - cur_score
                if d < 0 or np.random.rand() < np.exp(-d / T):
                    val = cur[idx]
                    delta = -2.0 * val
                    cur_ac = cur_ac.copy()
                    cur_ac[idx:idx+N] += 2.0 * delta * cur
                    cur_ac[idx] += delta * delta
                    cur[idx] = -val
                    cur_sum += delta
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_s = cur.copy()
