                    if gain > best_gain:
                        best_gain = gain
                        best_cands = [x]
                    elif gain == best_gain:
                        best_cands.append(x)
