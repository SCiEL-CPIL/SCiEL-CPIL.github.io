<<<<<<< SEARCH
_T = 2.0 * _SQRT3  # touching distance between unit hexagons centers
=======
_T = _SQRT3  # touching distance between unit hexagons centers
>>>>>>> REPLACE

Also, the Nelder-Mead might still converge to a local minimum with overlaps if the penalty is not high enough or if it gets stuck.
We should verify the final result in `construct()` and if it has overlaps, revert to the best valid configuration found during the process (or the initial one).

Let's add a validation step in `construct`.

