<<<<<<< SEARCH
def try_candidate(pos, relax_maxiter=10000, relax_ftol=1e-12, relax_gtol=1e-10):
        nonlocal best_energy, best_pos, start_time
        if time.time() - start_time > TIME_LIMIT - 10:
            return
        
        relaxed = _relax(pos, maxiter=relax_maxiter, ftol=relax_ftol, gtol=relax_gtol)
=======
def try_candidate(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
        nonlocal best_energy, best_pos, start_time
        if time.time() - start_time > TIME_LIMIT - 10:
            return
        
        relaxed = _relax(pos, maxiter=maxiter, ftol=ftol, gtol=gtol)
>>>>>>> REPLACE

