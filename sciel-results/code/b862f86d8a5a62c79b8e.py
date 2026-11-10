# Probe: inspect the baseline to learn the exact interface and data format.
import inspect
try:
    from autoresearch_suite.science_economics_expansion import auction_baseline
    print("SIGNATURE:", inspect.signature(auction_baseline))
    print("DOC:\n", inspect.getdoc(auction_baseline))
    print("SOURCE:\n", inspect.getsource(auction_baseline))
except Exception as e:
    print("IMPORT ERR:", repr(e))
