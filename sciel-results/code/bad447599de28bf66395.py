        if _remaining() > 20.0 and np.all(np.isfinite(res.x)):
            cand = res.x.reshape(64, 2)
            ok, Vc = _validate(cand)
