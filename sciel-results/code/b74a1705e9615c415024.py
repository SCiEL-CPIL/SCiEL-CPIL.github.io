                for c in candidates:
                    gain = cnt[c]
                    conflict = 0
                    for sel in code:
                        overlap = len(ball_sets[c] & ball_sets[sel])
                        conflict += overlap
