                # Calculate R
                R = set()
                for p in bp_i:
                    if cov_count[p] <= 2:
                        R.add(int(p))
                for p in bp_j:
                    if cov_count[p] <= 2:
                        R.add(int(p))
