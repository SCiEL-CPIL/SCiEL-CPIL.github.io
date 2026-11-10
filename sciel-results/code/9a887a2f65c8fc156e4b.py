            # Remove new areas
            for a in new_areas:
                area_counts[a] -= 1
                if area_counts[a] == 0:
                    del area_counts[a]
