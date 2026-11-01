def get_luw_params(k, g):
    # k must be a prime power for LUW.
    # If k is not a prime power, return None.
    if not is_prime_power(k):
        return None
    
    # Determine d
    # LUW CD(d, q) is bipartite.
    # Girth >= d + 5.
    # We need girth >= g.
    # If g is even, we need d + 5 >= g => d >= g - 5.
    # If g is odd, we need girth >= g+1 (since bipartite girth is even) => d + 5 >= g + 1 => d >= g - 4.
    
    if g % 2 == 0:
        d = g - 5
    else:
        d = g - 4
        
    d = max(3, d)
    return d
