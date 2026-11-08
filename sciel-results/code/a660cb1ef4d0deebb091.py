<<<<<<< SEARCH
    def random_search(rank, iters, seed, lr0=0.02, init_scale=0.3):
        """Randomized gradient search at a target rank; returns (U,V,W) if exact found."""
        if rank >= best_rank:
            return None
        rng = np.random.RandomState(seed)
        for it in range(iters):
            if now() > DEADLINE - 8:
                return None
            # Initialize: take baseline, truncate to `rank` columns with a random column subset, add noise
            cols = rng.choice(46, size=rank, replace=False)
            U = U_base[:, cols].astype(float).copy()
            V = V_base[:, cols].astype(float).copy()
            W = W_base[:, cols].astype(float).copy()
            # Small random perturbation
            U += rng.randn(*U.shape) * init_scale
            V += rng.randn(*V.shape) * init_scale
            W += rng.randn(*W.shape) * init_scale

            # Gradient descent with Adam-lite
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            lr = lr0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for t in range(60):
                if now() > DEADLINE - 6:
                    break
                prod = contract(U, V, W)
                diff = prod - T
                loss = float(np.sum(diff * diff))
                if loss < 1e-9:
                    U, V, W = snap(U), snap(V), snap(W)
                    if is_exact(U, V, W):
                        return (U, V, W)
                    break
                gU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                tt = t + 1
                mU = beta1*mU + (1-beta1)*gU; vU = beta2*vU + (1-beta2)*gU**2
                mV = beta1*mV + (1-beta1)*gV; vV = beta2*vV + (1-beta2)*gV**2
                mW = beta1*mW + (1-beta1)*gW; vW = beta2*vW + (1-beta2)*gW**2
                U -= lr*(mU/(1-beta1**tt))/(np.sqrt(vU/(1-beta2**tt))+eps)
                V -= lr*(mV/(1-beta1**tt))/(np.sqrt(vV/(1-beta2**tt))+eps)
                W -= lr*(mW/(1-beta1**tt))/(np.sqrt(vW/(1-beta2**tt))+eps)
                lr *= 0.995
            # Final snap check
            U2, V2, W2 = snap(U), snap(V), snap(W)
            if is_exact(U2, V2, W2):
                return (U2, V2, W2)
        return None
=======
    def random_search(rank, iters, seed, lr0=0.02, init_scale=0.3):
        """Randomized gradient search at a target rank; returns (U,V,W) if exact found."""
        if rank >= best_rank:
            return None
        rng = np.random.RandomState(seed)
        for it in range(iters):
            if now() > DEADLINE - 8:
                return None
            # Initialize: take baseline, truncate to `rank` columns with a random column subset, add noise
            cols = rng.choice(46, size=rank, replace=False)
            U = U_base[:, cols].astype(float).copy()
            V = V_base[:, cols].astype(float).copy()
            W = W_base[:, cols].astype(float).copy()
            # Small random perturbation
            U += rng.randn(*U.shape) * init_scale
            V += rng.randn(*V.shape) * init_scale
            W += rng.randn(*W.shape) * init_scale

            # Gradient descent with Adam-lite
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            lr = lr0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for t in range(80):
                if now() > DEADLINE - 6:
                    break
                prod = contract(U, V, W)
                diff = prod - T
                loss = float(np.sum(diff * diff))
                if loss < 1e-9:
                    U, V, W = snap(U), snap(V), snap(W)
                    if is_exact(U, V, W):
                        return (U, V, W)
                    break
                gU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                tt = t + 1
                mU = beta1*mU + (1-beta1)*gU; vU = beta2*vU + (1-beta2)*gU**2
                mV = beta1*mV + (1-beta1)*gV; vV = beta2*vV + (1-beta2)*gV**2
                mW = beta1*mW + (1-beta1)*gW; vW = beta2*vW + (1-beta2)*gW**2
                U -= lr*(mU/(1-beta1**tt))/(np.sqrt(vU/(1-beta2**tt))+eps)
                V -= lr*(mV/(1-beta1**tt))/(np.sqrt(vV/(1-beta2**tt))+eps)
                W -= lr*(mW/(1-beta1**tt))/(np.sqrt(vW/(1-beta2**tt))+eps)
                lr *= 0.995
            # Final snap check
            U2, V2, W2 = snap(U), snap(V), snap(W)
            if is_exact(U2, V2, W2):
                return (U2, V2, W2)
            # Try quarter-snap as well
            U3, V3, W3 = snap_to_quarter(U), snap_to_quarter(V), snap_to_quarter(W)
            if is_exact(U3, V3, W3):
                return (U3, V3, W3)
        return None
>>>>>>> REPLACE

