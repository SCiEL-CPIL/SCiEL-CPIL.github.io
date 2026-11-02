<<<<<<< SEARCH
    # LUW parameters
    # For odd girth g, we need d such that d + 5 >= g.
    # Smallest d is g - 5.
    # If g - 5 is even, we might be able to use a smaller d?
    # The girth of CD(d, q) is exactly d + 5 if d is odd? Or at least?
    # The paper states girth is at least d + 5.
    # To minimize vertices n = 2 * q^(d+1), we want smallest d.
    d = g - 5
    if d < 1:
        d = 1
=======
    # LUW parameters
    # The girth of the standard LUW construction CD(d, q) is 2d or 2d+1.
    # To guarantee girth >= g, we require 2d >= g.
    # Thus d = ceil(g / 2).
    # This ensures the graph is valid, though n = 2 * q^(d+1) might be large.
    d = (g + 1) // 2
    if d < 1:
        d = 1
>>>>>>> REPLACE
