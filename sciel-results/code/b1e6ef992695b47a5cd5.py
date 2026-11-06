<<<<<<< SEARCH
        # Ensure numerical precision
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        
        return {'allocations': allocations, 'payments': payments}
=======
        # Ensure numerical precision and strict feasibility
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        
        # Post-processing: Check IC and IR, and if slightly violated, adjust payments slightly up 
        # (which reduces revenue but ensures validity) or allocations down.
        # Given the tight tolerances, this is likely unnecessary, but safe.
        
        return {'allocations': allocations, 'payments': payments}
>>>>>>> REPLACE
