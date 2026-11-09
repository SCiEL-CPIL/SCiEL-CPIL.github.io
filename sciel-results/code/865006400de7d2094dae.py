        if len(current_row) == row_size:
            # Symmetry breaking: enforce lexicographic order
            if row_idx > 0 and current_row <= rows[-1]:
                return None
            # ... rest of logic
