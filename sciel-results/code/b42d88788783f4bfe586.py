def generate_m_sequence(m, poly):
    # poly is a list of bit positions for the feedback taps
    # e.g. for x^9 + x^4 + 1, taps are 9, 4, 0? 
    # Standard form: feedback from tap k to input.
    # Let's use a known primitive polynomial for m=9.
    # x^9 + x^4 + 1 is primitive? 
    # Let's just use a random LFSR with a known primitive poly.
    # Or simpler: Use a chirp sequence.
    # x_i = sign(sin(2*pi*i^2/N))?
    # Chirps have low sidelobes.
    
    # Let's try Chirp first. It's easier to code.
    # N=400.
    # t = np.arange(N)
    # x = np.sign(np.cos(2 * np.pi * t**2 / N))
    
    # Let's evaluate cost of Chirp vs Random.
    # Chirp autocorrelation is bounded.
    
    # I will use a Chirp as the initial state for SA.
    # This is a "different design" (Structured Start + SA) vs (Random Start + Greedy).
