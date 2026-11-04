<<<<<<< SEARCH
    # pick 11 centers: index 0 (center), 1..6 (ring1), and 4 from ring2 (indices 7..18)
    chosen = list(range(7)) + list(range(7, 11))  # center+ring1 + 4 ring2
=======
    # pick 11 centers: index 0 (center), 1..6 (ring1), and 4 from ring2 (indices 7..18)
    # We need to ensure we pick a symmetric subset of ring2 to maintain balance, or just any 4.
    # Indices 7..18 are ring2.
    chosen = list(range(7)) + list(range(7, 11))  # center+ring1 + 4 ring2
>>>>>>> REPLACE
