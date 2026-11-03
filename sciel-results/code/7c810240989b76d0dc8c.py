<<<<<<< SEARCH
    # scale by alpha (soft-min of radius ~ (1 - F/2)/alpha)
    return alpha * F, alpha * g
=======
    # scale by alpha (soft-min of radius ~ (1 - F/2)/alpha)
    # Return 1D gradient for minimize compatibility
    return alpha * F, (alpha * g).ravel()
>>>>>>> REPLACE
