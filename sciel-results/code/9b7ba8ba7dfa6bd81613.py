        C = np.zeros((N, N), dtype=np.uint8)
        powers = [q ** i for i in range(n)]
        
        for i in range(N):
            C[i, i] = 1
            if radius == 1:
                # Decode i to get digits? Or just iterate?
                # We need the digits to know what to change.
                # We can decode i into digits.
                w = list(words[i]) # We already have words list
                # Or decode from i
                # Using words[i] is fine since we generated it.
                
                for p in range(n):
                    old_val = w[p]
                    for new_val in range(q):
                        if new_val == old_val:
                            continue
                        # Calculate neighbor index
                        # idx = sum(w[j] * q^j)
                        # neighbor_idx = idx + (new_val - old_val) * q^p
                        neighbor_idx = i + (new_val - old_val) * powers[p]
                        C[i, neighbor_idx] = 1
            else:
                # Fallback
                w = words[i]
                for j in range(N):
                    w2 = words[j]
                    dist = sum(1 for a, b in zip(w, w2) if a != b)
                    if dist <= radius:
                        C[i, j] = 1
