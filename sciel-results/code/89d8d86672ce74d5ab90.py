<<<<<<< SEARCH
        # Add Laderman's 49 terms (for B columns 0-3)
        # Note: Laderman's B_index uses k*4+j, but we need k*5+j for the 4x4x5 case
        # So we need to adjust: in the 4x4x4 decomposition, B[k,j] has index k*4+j
        # In the 4x4x5 case, B[k,j] has index k*5+j
        # For j in 0..3, the mapping is: old_index = k*4+j -> new_index = k*5+j
        for col in range(49):
=======
        # Add the 4x4x4 terms (for B columns 0-3)
        # Note: the 4x4x4 B_index uses k*4+j, but we need k*5+j for the 4x4x5 case
        # For j in 0..3, the mapping is: old_index = k*4+j -> new_index = k*5+j
        for col in range(len(laderman_terms)):
>>>>>>> REPLACE

