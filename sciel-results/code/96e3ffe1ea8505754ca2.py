import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 112.0  # explicit safety margin well before the 150s kill

    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    Af = A.astype(np.float64)
    deg = Af.sum(axis=1)
    has_edge = deg > 0
    BIG = 1e9

    # ---- 1. Damped fixed-point "safe-to-pick" base score ----------------------
    # s[i] ~ 1 / (1 + soft-max of neighbor scores): a vertex is safer when its
    # neighbors are weak. soft-max via log-sum-exp is a smooth surrogate for the
    # hard max, which is better conditioned for damped iteration.
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG
    prev = s.copy()
    for it in range(300):
        if time.time() > deadline:
            break
        nbr = Af @ s
        m = np.max(nbr, axis=1)
        e = np.exp(np.clip(nbr - m[:, None], -700.0, 700.0))
        sm = m + np.log(e.sum(axis=1) + 1e-300)          # smooth max
        new_s = 1.0 / (1.0 + sm)
        new_s[~has_edge] = BIG
        s = 0.5 * s + 0.5 * new_s                          # damping
        if np.allclose(s, prev, rtol=1e-7, atol=1e-9):
            break
        prev = s.copy()

    base = s.copy()
    base[~has_edge] = 0.0                                  # isolate handling below

    # ---- 2. Spectral (Fiedler) periphery signal -------------------------------
    # Magnitude of the Fiedler vector marks vertices on the periphery of the main
    # component -> less central to connectivity -> safer to keep.
    spec = np.zeros(n)
    try:
        D = np.diag(deg)
        L = D - Af
        k = max(2, min(8, n))
        vals, vecs = np.linalg.eigsh(L, k=k, sigma=-1e-6)
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        spec = np.abs(fiedler)
    except Exception:
        # Fallback: power iteration on (I - D^-1 A) to approximate the
        # second-smallest Laplacian mode (slowest non-trivial decay).
        try:
            invD = np.zeros_like(deg)
            invD[deg > 0] = 1.0 / deg[deg > 0]
            M = np.eye(n) - Af * invD[:, None]
            v = np.ones(n) / np.sqrt(n)
            for _ in range(300):
                if time.time() > deadline:
                    break
                v = M @ v
                v -= np.mean(v)
                nn = np.linalg.norm(v)
                if nn < 1e-15:
                    break
                v /= nn
            spec = np.abs(v)
        except Exception:
            spec = np.zeros(n)

    # ---- 3. Combine base + spectral with adaptive weighting -------------------
    # Boost the spectral signal when base scores are tightly clustered (little
    # signal to differentiate), so the periphery ordering actually matters.
    if base.max() > base.min():
        spread = (base.max() - base.min()) / (base.mean() + 1e-12)
        w_spec = min(1.0, 0.3 + 0.7 * (1.0 - min(1.0, spread)))
    else:
        w_spec = 1.0

    base_norm = (base - base.min()) / (base.max() - base.min() + 1e-12)
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-12)
    combined = (1.0 - w_spec) * base_norm + w_spec * spec_norm

    # ---- 4. Neighborhood refinement: penalize contested (high-degree) vertices
    scores = combined - 0.05 * (deg / (deg.max() + 1.0))

    # ---- 5. Isolated vertices always go on top --------------------------------
    scores[~has_edge] = BIG

    return scores
