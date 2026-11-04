                gaps = np.array(gaps)
                # Smooth max
                max_gap = np.max(gaps)
                smooth_max = max_gap + (1.0 / 10.0) * np.log(np.sum(np.exp(10.0 * (gaps - max_gap))))
                if smooth_max < 0:
                    penalty += (-smooth_max)**2
