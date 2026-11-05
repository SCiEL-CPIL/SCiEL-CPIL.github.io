                    colors[pos] = new_color
                    new_viol = len(set(colors[p] for p in ap)) < 3
                    colors[pos] = old_color
