import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()


def _autocorr_fast(h):
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _hard_score(h):
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _flip_ac(h, ac, i):
    """Compute new autocorrelation after flipping h[i], in O(N)."""
    ac_new = ac.copy()
    hi = h[i]
    k = np.arange(1, N)
    vp = (i + k) < N
    hp = np.zeros(N - 1)
    if np.any(vp):
        hp[vp] = h[i + k[vp]]
    vm = (i - k) >= 0
    hm = np.zeros(N - 1)
    if np.any(vm):
        hm[vm] = h[i - k[vm]]
    ac_new[1:] += -2.0 * hi * (hp + hm)
    return ac_new


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 110.0  # Safety margin
        
        rng = np.random.default_rng(42)
        idx = np.arange(N)
        
        # Generate diverse initial candidates
        inits = []
        # Thue-Morse
        t = [1]
        while len(t) < N:
            t += [-x for x in t]
        inits.append(np.array(t[:N]))
        
        # Random
        for _ in range(5):
            inits.append(rng.choice(np.array([-1.0, 1.0]), size=N))
            
        # Chirps / Quadratic phases
        inits.append(np.sign(np.cos(2 * np.pi * idx ** 2 / (2 * N))))
        inits.append(np.sign(np.cos(2 * np.pi * idx ** 2 / (N + 1))))
        inits.append(np.sign(np.cos(2 * np.pi * 2 * idx / N)))
        inits.append(np.sign(np.sin(2 * np.pi * 3 * idx / N)))

        best_h = None
        best_hard = 1e9
        
        for h in inits:
            h = np.sign(h)
            if np.all(h == 0):
                h = np.ones(N)
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        # Simulated Annealing with Restart
        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr_fast(h)
        
        T = 0.5
        T_min = 0.001
        cooling = 0.995
        max_sa_steps = 20000
        stagnation_limit = 1000
        no_improve_count = 0
        
        for step in range(max_sa_steps):
            if time.time() > deadline:
                break
            
            # Choose move: 80% single flip, 20% block flip (size 2-4)
            if rng.random() < 0.8:
                i = int(rng.integers(0, N))
                ac_new = _flip_ac(h, ac, i)
                h[i] *= -1.0
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                
                if new_hard < cur_hard or rng.random() < np.exp(-(new_hard - cur_hard) / T):
                    cur_hard = new_hard
                    ac = ac_new
                    if cur_hard < best_hard:
                        best_hard = cur_hard
                        best_h = h.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    h[i] *= -1.0 # revert
                    no_improve_count += 1
            else:
                # Block flip
                bs = int(rng.integers(2, 5))
                flip_idx = rng.choice(N, size=bs, replace=False)
                h[flip_idx] *= -1.0
                ac_new = _autocorr_fast(h)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                
                if new_hard < cur_hard or rng.random() < np.exp(-(new_hard - cur_hard) / T):
                    cur_hard = new_hard
                    ac = ac_new
                    if cur_hard < best_hard:
                        best_hard = cur_hard
                        best_h = h.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    h[flip_idx] *= -1.0 # revert
                    no_improve_count += 1
            
            T = max(T_min, T * cooling)
            
            # Restart if stuck
            if no_improve_count > stagnation_limit:
                # Perturb best_h and restart
                h = best_h.copy()
                # Randomly flip 5% of bits
                pert_idx = rng.choice(N, size=int(0.05 * N), replace=False)
                h[pert_idx] *= -1.0
                cur_hard = _hard_score(h)
                ac = _autocorr_fast(h)
                T = 0.5
                no_improve_count = 0
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()

        # Final Polish: Greedy single-flip descent
        h = best_h.copy()
        for _pass in range(20):
            if time.time() > deadline - 5.0:
                break
            any_improve = False
            ac = _autocorr_fast(h)
            dc = ac[0]
            if dc <= 1e-30:
                break
            cur_peak = float(np.max(np.abs(ac)) / dc)
            for i in range(N):
                ac_new = _flip_ac(h, ac, i)
                new_peak = float(np.max(np.abs(ac_new)) / ac_new[0])
                if new_peak < cur_peak - 1e-14:
                    h[i] *= -1.0
                    ac = ac_new
                    cur_peak = new_peak
                    any_improve = True
            if not any_improve:
                break
            if cur_peak < best_hard:
                best_hard = cur_peak
                best_h = h.copy()

        final_hard = _hard_score(h)
        if final_hard < best_hard:
            best_hard = final_hard
            best_h = h.copy()

        return best_h
    except Exception:
        return np.ones(N)
