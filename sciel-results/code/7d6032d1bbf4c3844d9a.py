        if len(reachable) > 1000000:
             # Prune or break?
             # If we break, we might miss a solution.
             # But if the solution has a large max denominator, the seed might be better?
             # No, seed is likely huge.
             # Let's just let it run, but monitor time.
             pass
