            for pos in range(len(seq) + 1):
                # Check sorted order
                if pos > 0 and not (seq[pos-1] < c):
                    continue
                if pos < len(seq) and not (c < seq[pos]):
                    continue
                # ... rest of logic
