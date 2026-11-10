    def get_max_discrepancy_info(s):
        n_local = len(s)
        max_abs = 0
        critical_indices = np.zeros(n_local, dtype=bool)
        
        for step in range(1, n_local + 1):
            for r in range(step):
                subseq = s[r::step]
                if len(subseq) == 0:
                    continue
                
                # Find max abs subarray sum and its location
                max_ending = 0
                max_sofar = 0
                min_ending = 0
                min_sofar = 0
                
                # To track indices, we need to store the start/end of the max subarray
                # This makes Kadane's slightly more complex.
                # For now, let's just get the value.
                # If we find a value == max_abs, we need to mark indices.
                
                # Let's just compute the value first.
                # We can re-scan if it matches max_abs to get indices.
                
                # Optimization: Just compute value.
                for x in subseq:
                    if max_ending < 0:
                        max_ending = x
                    else:
                        max_ending += x
                    if max_sofar < max_ending:
                        max_sofar = max_ending
                    
                    if min_ending > 0:
                        min_ending = x
                    else:
                        min_ending += x
                    if min_sofar > min_ending:
                        min_sofar = min_ending
                
                local_max = max(max_sofar, -min_sofar)
                
                if local_max > max_abs:
                    max_abs = local_max
                    # Reset critical indices? No, we need to find them.
                    # We can clear and re-mark.
                    critical_indices = np.zeros(n_local, dtype=bool)
                    # We need to find WHICH subarray gave this max.
                    # Re-run Kadane to find indices?
                    # Or just mark all indices in this subseq?
                    # That's an over-approximation, but safe.
                    # Indices in subseq are r, r+step, ...
                    critical_indices[r::step] = True
                    
                elif local_max == max_abs and max_abs > 0:
                    # Add indices of this subseq to critical set
                    # Only if we want to be precise. 
                    # For efficiency, just add the whole subseq indices.
                    critical_indices[r::step] = True

        return max_abs, critical_indices
