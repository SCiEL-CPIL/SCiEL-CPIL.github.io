      def compute_can_add(S_mask, line_mates):
          S_expanded = S_mask[line_mates]  # (2401, 50, 48)
          covered = S_expanded.any(axis=2)  # (2401, 50) -> True if any mate in S
          return covered.all(axis=1)  # True if ALL directions have a mate in S
      