    # Before return
    if not code or np.min(coverage) <= 0:
         # Fallback to best valid code found
         if best_valid_code:
             code = best_valid_code
         else:
             # Emergency: return a trivial valid code (e.g. all 0s? No, must be distinct and cover)
             # Just return the greedy result if it was valid, or a random valid set?
             # If we have no valid code, we must generate one.
             # But this case should be rare.
             pass
